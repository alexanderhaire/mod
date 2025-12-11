import calendar
import datetime
import re
from decimal import Decimal
from pathlib import Path

import pyodbc

from bom_guidance import build_bom_guidance
from constants import CUSTOM_SQL_ALLOWED_TABLES, CUSTOM_SQL_MAX_ROWS, LOGGER, PRIMARY_LOCATION, SCHEMA_PRIORITY_COLUMNS
from context_utils import summarize_sql_context
from vendor_analytics import handle_vendor_scorecard
from inventory_queries import (
    fetch_mfg_bom_grouped_by_component,
    fetch_on_hand_by_item,
    fetch_open_po_supply,
    fetch_open_po_supply,
    fetch_recursive_bom_for_item,
)
from openai_clients import call_openai_sql_generator
from parsing_utils import (
    decimal_or_zero,
    extract_item_from_prompt,
    extract_lot_from_prompt,
    normalize_item_for_bom,
    parse_month_year_from_prompt,
    parse_percent_increase,
)
from schema_utils import load_allowed_sql_schema, summarize_schema_for_prompt
from sql_utils import (
    extract_invalid_column_names,
    extract_table_tokens,
    format_sql_preview,
    normalize_sql_params,
    summarize_table_columns,
    validate_custom_sql,
)
from dynamic_handler import find_dynamic_handler, save_dynamic_handler
from training_logger import record_sql_example, should_log_training_example

USE_DYNAMIC_HANDLERS = False  # Disable learned handlers to keep responses model-driven.
_MAX_CHART_ROWS_FOR_IMAGE = 500


def _log_sql_result(label: str, sql_preview: str | None, rows: list | None, truncated: bool, handler_name: str | None = None) -> None:
    """Log executed SQL with a compact preview to aid debugging."""
    row_count = len(rows) if isinstance(rows, list) else 0
    preview = (sql_preview or "").replace("\n", " ").strip()
    if len(preview) > 500:
        preview = preview[:500] + "..."
    name = f"{label}:{handler_name}" if handler_name else label
    LOGGER.info("SQL %s returned %s row(s)%s | %s", name, row_count, " (truncated)" if truncated else "", preview or "<no preview>")


_MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "febuary": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _pick_schema_column(schema: dict[str, list[dict]], table: str, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column (case-insensitive) from a schema snapshot."""
    if not schema or not candidates:
        return None
    available = {col["name"].lower(): col["name"] for col in schema.get(table, []) if isinstance(col, dict)}
    for candidate in candidates:
        if not candidate:
            continue
        found = available.get(candidate.lower())
        if found:
            return found
    return None


def _extract_months_from_prompt(prompt: str) -> list[int]:
    """Return month numbers in prompt order, handling common abbreviations/misspellings."""
    pattern = r"\b(" + "|".join(_MONTH_MAP.keys()) + r")\b"
    seen: list[int] = []
    for match in re.finditer(pattern, prompt.lower()):
        month_num = _MONTH_MAP.get(match.group(1))
        if month_num and month_num not in seen:
            seen.append(month_num)
    return seen


def _extract_year_from_prompt(prompt: str) -> int | None:
    """Find a four-digit year in the prompt, if present."""
    match = re.search(r"\b(19|20)\d{2}\b", prompt)
    return int(match.group(0)) if match else None


def _extract_month_year_pairs(prompt: str, today: datetime.date) -> list[tuple[int, int]]:
    """
    Return (year, month) pairs in prompt order, pairing years to months using nearby year tokens
    and reasonable propagation across year boundaries (e.g., 'December 2023 ... April 2024').
    """
    lower = prompt.lower()
    pattern = r"(?P<month>" + "|".join(_MONTH_MAP.keys()) + r")\s*(?P<year>(?:19|20)\d{2})?"
    month_matches = []
    used_spans: list[tuple[int, int]] = []
    for m in re.finditer(pattern, lower):
        month_num = _MONTH_MAP.get(m.group("month"))
        if not month_num:
            continue
        year_val = int(m.group("year")) if m.group("year") else None
        month_matches.append({"month": month_num, "year": year_val, "pos": m.start(), "end": m.end()})
        if year_val is not None:
            used_spans.append((m.start("year"), m.end("year")))

    if not month_matches:
        return []

    year_tokens = []
    for y in re.finditer(r"(19|20)\d{2}", lower):
        span = (y.start(), y.end())
        if any(not (span[1] <= us[0] or span[0] >= us[1]) for us in used_spans):
            continue
        year_tokens.append((int(y.group()), y.start()))
    year_tokens.sort(key=lambda t: t[1])

    pairs: list[tuple[int, int]] = []
    current_year = None
    for match in month_matches:
        if match["year"]:
            current_year = match["year"]
            pairs.append((current_year, match["month"]))
            continue

        if current_year is None and year_tokens:
            # Pick the nearest future year token if available; otherwise the last year token.
            future_year = next((yt[0] for yt in year_tokens if yt[1] >= match["pos"]), None)
            current_year = future_year if future_year else year_tokens[-1][0]
        if current_year is None:
            current_year = today.year

        pairs.append((current_year, match["month"]))

    deduped: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for pair in pairs:
        if pair in seen_pairs:
            LOGGER.debug("Dropping overlapping month/year pair %s from prompt '%s'", pair, prompt)
            continue
        seen_pairs.add(pair)
        deduped.append(pair)

    return deduped


def handle_top_selling_question(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """Deterministic handler for 'top selling items' questions that mention a month (and optional year)."""
    top_keywords = ("top selling", "top sales", "top items", "top sellers", "most used")
    if not any(k in prompt.lower() for k in top_keywords):
        return None
    month, year = parse_month_year_from_prompt(prompt, today, prefer_same_year=True)
    if month is None or year is None:
        return None

    requested_year = year
    target_period_start = datetime.date(year, month, 1)
    is_future_period = target_period_start > today
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"

    query = """
        SELECT TOP 200
            l.ITEMNMBR,
            i.ITEMDESC,
            SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS TotalSales
        FROM SOP30300 l
        JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        JOIN IV00101 i ON l.ITEMNMBR = i.ITEMNMBR
        WHERE h.DOCDATE BETWEEN ? AND ?
          AND l.SOPTYPE IN (3, 4) -- Invoice / Return
        GROUP BY l.ITEMNMBR, i.ITEMDESC
        ORDER BY TotalSales DESC
    """
    sql_preview = None
    try:
        cursor.execute(query, start_date, end_date)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
        sql_preview = format_sql_preview(query, [start_date, end_date])
    except pyodbc.Error as err:
        return {"error": f"Failed to query top selling items: {err}"}

    fallback_note = None
    if not rows and is_future_period:
        fallback_year = requested_year - 1
        start_date = f"{fallback_year}-{month:02d}-01"
        end_date = f"{fallback_year}-{month:02d}-{calendar.monthrange(fallback_year, month)[1]}"
        try:
            cursor.execute(query, start_date, end_date)
            fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
            columns = [c[0] for c in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
            truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
            sql_preview = format_sql_preview(query, [start_date, end_date])
            year = fallback_year
            fallback_note = (
                f"Requested {calendar.month_name[month]} {requested_year} is in the future; "
                f"showing {calendar.month_name[month]} {fallback_year} history instead."
            )
        except pyodbc.Error as err:
            return {"error": f"Failed to query top selling items fallback: {err}"}

    if not rows:
        return {"insights": {"summary": f"No shipped invoice history found for {calendar.month_name[month]} {year}."}}

    summary = (
        f"Top selling items for {calendar.month_name[month]} {year}, based on shipped invoices "
        f"from SOP history with returns netted."
    )
    insights = {"summary": summary, "row_count": len(rows), "truncated": truncated}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows."
    if fallback_note:
        insights["note"] = f"{fallback_note}" if "note" not in insights else f"{insights['note']}\n\n{fallback_note}"

    entities = {"month": month, "year": year, "intent": "sales"}
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_raw_material_usage_overall(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Rank raw materials by total usage across a recent window (or all time when requested).
    Captures prompts like 'what raw materials have the highest usage?' that lack a specific month.
    """
    lower = prompt.lower()
    usage_tokens = ("use", "used", "usage", "consume", "consumed", "consumption")
    raw_tokens = ("raw material", "raw materials", "materials", "ingredients", "components")
    ranking_tokens = ("most", "top", "highest", "biggest", "largest")

    months_mentioned = _extract_months_from_prompt(prompt)
    if months_mentioned:
        return None
    if not any(tok in lower for tok in usage_tokens):
        return None
    if not (any(tok in lower for tok in raw_tokens) or any(tok in lower for tok in ranking_tokens)):
        return None

    recent_tokens = ("last 12", "past year", "last year", "recent", "recently")
    all_time_tokens = ("all time", "overall", "ever", "lifetime")
    use_recent_window = any(tok in lower for tok in recent_tokens)
    if any(tok in lower for tok in all_time_tokens):
        use_recent_window = False
    params: list[str] = []
    date_filter = ""
    note_parts: list[str] = []
    if use_recent_window:
        start_date = today - datetime.timedelta(days=364)
        end_date = today
        date_filter = "AND h.DOCDATE BETWEEN ? AND ?"
        params.extend([start_date.isoformat(), end_date.isoformat()])
        note_parts.append(f"Lookback: last 365 days through {end_date.isoformat()}.")

    query = f"""
        SELECT TOP 200
            t.ITEMNMBR,
            i.ITEMDESC,
            SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQty
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        LEFT JOIN IV00101 i ON i.ITEMNMBR = t.ITEMNMBR
        WHERE 1=1
            {date_filter}
        GROUP BY t.ITEMNMBR, i.ITEMDESC
        HAVING SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) > 0
        ORDER BY UsageQty DESC
    """
    sql_preview = None
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
        sql_preview = format_sql_preview(query, params)
    except pyodbc.Error as err:
        return {"error": f"Failed to query raw-material usage ranking: {err}"}

    if not rows:
        return {"insights": {"summary": "No raw-material consumption found for the selected window."}}

    summary = "Top raw-material consumption by total usage"
    if use_recent_window:
        summary += " (last 12 months)"
    insights = {"summary": summary, "row_count": len(rows)}
    note = " ".join(note_parts).strip()
    if note:
        insights["note"] = note
    if truncated:
        insights["note"] = f"{insights.get('note', '')} Showing first {CUSTOM_SQL_MAX_ROWS} rows.".strip()

    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": {"intent": "usage"}}


def handle_raw_material_usage_multi_month(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Rank the top raw materials for multiple specific months in one shot.
    Example: 'What raw materials had the highest usage in December, January, February, March, and April of 2024?'
    """
    lower = prompt.lower()
    usage_tokens = ("use", "used", "usage", "consume", "consumed", "consumption")
    raw_tokens = ("raw material", "raw materials", "materials", "ingredients", "components")
    ranking_tokens = ("most", "top", "highest", "biggest", "largest")

    pairs = _extract_month_year_pairs(prompt, today)
    if len(pairs) < 2:
        return None
    if not any(tok in lower for tok in usage_tokens):
        return None
    if not (any(tok in lower for tok in raw_tokens) or any(tok in lower for tok in ranking_tokens)):
        return None

    period_filters = " OR ".join("(YEAR(h.DOCDATE) = ? AND MONTH(h.DOCDATE) = ?)" for _ in pairs)
    params: list[int | str] = []
    for yr, mo in pairs:
        params.extend([yr, mo])

    top_per_month = 5
    params.append(top_per_month)

    query = f"""
        WITH usage AS (
            SELECT
                YEAR(h.DOCDATE) AS Year,
                MONTH(h.DOCDATE) AS Month,
                t.ITEMNMBR,
                i.ITEMDESC,
                SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQty
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            LEFT JOIN IV00101 i ON i.ITEMNMBR = t.ITEMNMBR
            WHERE ({period_filters})
            GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE), t.ITEMNMBR, i.ITEMDESC
            HAVING SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) > 0
        ),
        ranked AS (
            SELECT
                Year,
                Month,
                ITEMNMBR,
                ITEMDESC,
                UsageQty,
                ROW_NUMBER() OVER (PARTITION BY Year, Month ORDER BY UsageQty DESC) AS rn
            FROM usage
        )
        SELECT Year, Month, ITEMNMBR, ITEMDESC, UsageQty
        FROM ranked
        WHERE rn <= ?
        ORDER BY Year, Month, UsageQty DESC
    """
    sql_preview = None
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
        sql_preview = format_sql_preview(query, params)
    except pyodbc.Error as err:
        return {"error": f"Failed to query raw-material usage by month: {err}"}

    if not rows:
        month_names = ", ".join(f"{calendar.month_name[m]} {y}" for y, m in pairs)
        return {"insights": {"summary": f"No raw-material consumption found for {month_names}."}}

    month_names = ", ".join(f"{calendar.month_name[m]} {y}" for y, m in pairs)
    summary = f"Top {top_per_month} raw-materials by usage for {month_names} (negative TRXQTY summed as positive)."
    insights = {"summary": summary, "row_count": len(rows)}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows across all months."

    entity_years = sorted({y for y, _ in pairs})
    entity_months = [m for _, m in pairs]
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": {"years": entity_years, "months": entity_months, "intent": "usage"}}


def handle_raw_material_usage_multi_month_total(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Sum raw-material usage across multiple months and return the top items by combined usage.
    Example: 'Can you show the usage as a sum of the usage in those months?'
    """
    lower = prompt.lower()
    usage_tokens = ("use", "used", "usage", "consume", "consumed", "consumption")
    raw_tokens = ("raw material", "raw materials", "materials", "ingredients", "components")
    sum_tokens = ("sum", "total", "combined", "aggregate", "rollup")

    pairs = _extract_month_year_pairs(prompt, today)
    if len(pairs) < 2:
        return None
    if not any(tok in lower for tok in usage_tokens):
        return None
    if not (any(tok in lower for tok in raw_tokens) or any(tok in lower for tok in sum_tokens)):
        return None
    if not any(tok in lower for tok in sum_tokens):
        return None

    period_filters = " OR ".join("(YEAR(h.DOCDATE) = ? AND MONTH(h.DOCDATE) = ?)" for _ in pairs)
    params: list[int | str] = []
    for yr, mo in pairs:
        params.extend([yr, mo])
    top_items = 200
    params.append(top_items)

    query = f"""
        WITH usage AS (
            SELECT
                t.ITEMNMBR,
                i.ITEMDESC,
                SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQty
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            LEFT JOIN IV00101 i ON i.ITEMNMBR = t.ITEMNMBR
            WHERE ({period_filters})
            GROUP BY t.ITEMNMBR, i.ITEMDESC
            HAVING SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) > 0
        ),
        ranked AS (
            SELECT
                ITEMNMBR,
                ITEMDESC,
                UsageQty AS TotalUsage,
                ROW_NUMBER() OVER (ORDER BY UsageQty DESC) AS rn
            FROM usage
        )
        SELECT ITEMNMBR, ITEMDESC, TotalUsage
        FROM ranked
        WHERE rn <= ?
        ORDER BY TotalUsage DESC
    """
    sql_preview = None
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
        sql_preview = format_sql_preview(query, params)
    except pyodbc.Error as err:
        return {"error": f"Failed to sum raw-material usage across months: {err}"}

    if not rows:
        month_names = ", ".join(f"{calendar.month_name[m]} {y}" for y, m in pairs)
        return {"insights": {"summary": f"No raw-material consumption found across {month_names}."}}

    month_names = ", ".join(f"{calendar.month_name[m]} {y}" for y, m in pairs)
    summary = f"Total raw-material usage across {month_names} (negative TRXQTY summed as positive)."
    insights = {"summary": summary, "row_count": len(rows)}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows."

    entity_years = sorted({y for y, _ in pairs})
    entity_months = [m for _, m in pairs]
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": {"years": entity_years, "months": entity_months, "intent": "usage"}}


def handle_raw_material_usage_month(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Rank raw materials by usage for a given month using IV consumption history (negative TRXQTY).
    Designed to catch prompts like 'what raw materials do we use the most of in December?' without LLM routing,
    including short forms like 'top used items in January' when ranking and month cues are present.
    """
    lower = prompt.lower()
    usage_tokens = ("use", "used", "usage", "consume", "consumed", "consumption")
    raw_tokens = ("raw material", "raw materials", "materials", "ingredients", "components")
    ranking_tokens = ("most", "top", "highest", "biggest", "largest")

    has_usage = any(tok in lower for tok in usage_tokens)
    has_raw_hint = any(tok in lower for tok in raw_tokens)
    has_ranking = any(tok in lower for tok in ranking_tokens)
    has_item_words = any(tok in lower for tok in ("item", "items", "product", "products"))

    if not has_usage:
        return None
    if not has_raw_hint and not (has_ranking and has_item_words):
        return None
    if not has_ranking:
        return None

    month, year = parse_month_year_from_prompt(prompt, today, preference="past")
    if month is None or year is None:
        return None

    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"

    query = """
        SELECT TOP 200
            t.ITEMNMBR,
            i.ITEMDESC,
            SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQty
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        LEFT JOIN IV00101 i ON i.ITEMNMBR = t.ITEMNMBR
        WHERE h.DOCDATE BETWEEN ? AND ?
        GROUP BY t.ITEMNMBR, i.ITEMDESC
        HAVING SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) > 0
        ORDER BY UsageQty DESC
    """
    sql_preview = None
    try:
        cursor.execute(query, start_date, end_date)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
        sql_preview = format_sql_preview(query, [start_date, end_date])
    except pyodbc.Error as err:
        return {"error": f"Failed to query raw-material usage: {err}"}

    if not rows:
        return {"insights": {"summary": f"No raw-material consumption found for {calendar.month_name[month]} {year}."}}

    summary = (
        f"Top raw-material consumption for {calendar.month_name[month]} {year} "
        f"using IV history (negative TRXQTY summed as positive usage)."
    )
    insights = {"summary": summary, "row_count": len(rows)}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows."

    entities = {"month": month, "year": year, "intent": "usage"}
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_order_point_gap(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Rank items by the gap to their (seasonally adjusted) order point using availability, on-order supply,
    and recent/seasonal usage to suggest what to buy next.
    """
    lower = prompt.lower()
    order_point_tokens = (
        "order point",
        "order-point",
        "orderpoint",
        "order up to",
        "order-up-to",
        "orderupto",
        "gap to order point",
        "below order point",
        "order point qty",
        "reorder point",
    )
    if not any(tok in lower for tok in order_point_tokens):
        return None

    schema = load_allowed_sql_schema(cursor)
    if not schema:
        return {"error": "Schema metadata is unavailable; cannot calculate order-point gaps."}

    order_point_col = _pick_schema_column(schema, "IV00102", ("ORDERPOINTQTY", "REORDPRL", "REORDERPOINTQTY", "ORDERPOINT", "REORDPNTQTY", "REORDPNT", "ORDRPTQTY"))
    order_up_to_col = _pick_schema_column(schema, "IV00102", ("ORDERUPTOQTY", "ORDRUPTQTY", "ORDERUPTO", "ORDRUPTO", "ORDERUPTOLEVEL"))
    qty_available_col = _pick_schema_column(schema, "IV00102", ("QTYAVAIL", "QTYAVAILABLE", "QTYAVL", "QTYAVLBL", "QTYAVLB", "QTY_AVAIL"))
    qty_on_hand_col = _pick_schema_column(schema, "IV00102", ("QTYONHND", "QTYONHAND", "QTY_ON_HAND"))
    qty_on_order_col = _pick_schema_column(schema, "IV00102", ("QTYONPO", "QTYONPCH", "QTYONORDER", "QTYONORD", "QTY_ON_ORDER"))
    location_col = _pick_schema_column(schema, "IV00102", ("LOCNCODE", "LOCATIONCODE", "LOCN_CODE")) or "LOCNCODE"
    item_class_col = _pick_schema_column(schema, "IV00102", ("ITMCLSCD", "ITEMCLASS", "ITEMCLASSCODE", "ITEMCLSCD"))
    primary_location = PRIMARY_LOCATION

    if not order_point_col or not qty_on_hand_col:
        available_cols = summarize_table_columns({"IV00102"}, schema)
        return {
            "error": (
                "IV00102 is missing expected columns for order point analysis. "
                f"Tried order point columns and on-hand columns; available columns: {available_cols or 'unknown'}."
            )
        }

    order_point_expr = f"COALESCE(q.{order_point_col}, 0)"
    order_up_to_expr = f"COALESCE(q.{order_up_to_col}, q.{order_point_col}, 0)" if order_up_to_col else order_point_expr
    qty_on_hand_expr = f"COALESCE(q.{qty_on_hand_col}, 0)"
    qty_available_expr = f"COALESCE(q.{qty_available_col}, {qty_on_hand_expr})" if qty_available_col else qty_on_hand_expr
    qty_on_order_expr = f"COALESCE(q.{qty_on_order_col}, 0)" if qty_on_order_col else "0"

    recent_days = 90
    recent_start = today - datetime.timedelta(days=recent_days - 1)
    recent_end = today
    seasonal_year = today.year - 1
    seasonal_start = datetime.date(seasonal_year, today.month, 1)
    seasonal_end = datetime.date(seasonal_year, today.month, calendar.monthrange(seasonal_year, today.month)[1])
    seasonal_days = (seasonal_end - seasonal_start).days + 1
    projection_days = 30

    sql = f"""
        WITH Base AS (
            SELECT
                q.ITEMNMBR,
                i.ITEMDESC,
                q.{location_col} AS LocationCode,
                {order_point_expr} AS OrderPointQty,
                {order_up_to_expr} AS OrderUpToQty,
                {qty_available_expr} AS QtyAvailable,
                {qty_on_order_expr} AS QtyOnOrder,
                {qty_on_hand_expr} AS QtyOnHand
                {" , q." + item_class_col + " AS ItemClassCode" if item_class_col else ""}
            FROM IV00102 q
            LEFT JOIN IV00101 i ON i.ITEMNMBR = q.ITEMNMBR
            WHERE q.{location_col} = ?
        ),
        RecentUsage AS (
            SELECT
                t.ITEMNMBR,
                SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQty90D
            FROM IV30300 t
            INNER JOIN IV30200 h ON h.DOCNUMBR = t.DOCNUMBR AND h.IVDOCTYP = t.DOCTYPE
            WHERE h.DOCDATE BETWEEN ? AND ?
            GROUP BY t.ITEMNMBR
        ),
        SeasonalUsage AS (
            SELECT
                t.ITEMNMBR,
                SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQtySeason
            FROM IV30300 t
            INNER JOIN IV30200 h ON h.DOCNUMBR = t.DOCNUMBR AND h.IVDOCTYP = t.DOCTYPE
            WHERE h.DOCDATE BETWEEN ? AND ?
            GROUP BY t.ITEMNMBR
        )
        SELECT TOP {CUSTOM_SQL_MAX_ROWS}
            b.ITEMNMBR,
            b.ITEMDESC,
            b.LocationCode,
            b.OrderPointQty,
            b.OrderUpToQty,
            b.QtyAvailable,
            b.QtyOnOrder,
            b.QtyOnHand,
            ISNULL(r.UsageQty90D, 0) AS UsageLast{recent_days}D,
            ISNULL(s.UsageQtySeason, 0) AS UsageSameMonthLastYear,
            calc.AvgDailyUse90D,
            calc.SeasonalAvgDailyUse,
            calc.SeasonalityFactor,
            calc.Projected30DayDemand,
            calc.AdjustedOrderPoint,
            calc.GapToOrderPoint,
            calc.BuyToOrderUpTo
            {" , b.ItemClassCode" if item_class_col else ""}
        FROM Base b
        LEFT JOIN RecentUsage r ON r.ITEMNMBR = b.ITEMNMBR
        LEFT JOIN SeasonalUsage s ON s.ITEMNMBR = b.ITEMNMBR
        CROSS APPLY (
            SELECT
                CAST(ISNULL(r.UsageQty90D, 0) / {recent_days} AS DECIMAL(38, 6)) AS AvgDailyUse90D,
                CAST(ISNULL(s.UsageQtySeason, 0) / {seasonal_days} AS DECIMAL(38, 6)) AS SeasonalAvgDailyUse
        ) AS u
        CROSS APPLY (
            SELECT
                CASE
                    WHEN u.AvgDailyUse90D = 0 AND u.SeasonalAvgDailyUse = 0 THEN CAST(1.0 AS DECIMAL(38, 6))
                    WHEN u.AvgDailyUse90D = 0 THEN CAST(1.2 AS DECIMAL(38, 6))
                    WHEN u.SeasonalAvgDailyUse / NULLIF(u.AvgDailyUse90D, 0) > 2 THEN CAST(2.0 AS DECIMAL(38, 6))
                    WHEN u.SeasonalAvgDailyUse / NULLIF(u.AvgDailyUse90D, 0) < 0.5 THEN CAST(0.5 AS DECIMAL(38, 6))
                    ELSE CAST(u.SeasonalAvgDailyUse / NULLIF(u.AvgDailyUse90D, 0) AS DECIMAL(38, 6))
                END AS SeasonalityFactor
        ) AS f
        CROSS APPLY (
            SELECT
                u.AvgDailyUse90D,
                u.SeasonalAvgDailyUse,
                f.SeasonalityFactor,
                (u.AvgDailyUse90D * {projection_days} * f.SeasonalityFactor) AS Projected30DayDemand,
                (b.OrderPointQty * f.SeasonalityFactor) AS AdjustedOrderPoint,
                CASE
                    WHEN (b.OrderPointQty * f.SeasonalityFactor) - (b.QtyAvailable + b.QtyOnOrder) < 0 THEN 0
                    ELSE (b.OrderPointQty * f.SeasonalityFactor) - (b.QtyAvailable + b.QtyOnOrder)
                END AS GapToOrderPoint,
                CASE
                    WHEN (b.OrderUpToQty - (b.QtyAvailable + b.QtyOnOrder)) < 0 THEN 0
                    ELSE (b.OrderUpToQty - (b.QtyAvailable + b.QtyOnOrder))
                END AS BuyToOrderUpTo
        ) AS calc
        WHERE calc.GapToOrderPoint > 0 OR calc.BuyToOrderUpTo > 0
        ORDER BY calc.GapToOrderPoint DESC
    """

    sql_params = [
        primary_location,
        recent_start.isoformat(),
        recent_end.isoformat(),
        seasonal_start.isoformat(),
        seasonal_end.isoformat(),
    ]
    sql_preview = format_sql_preview(sql, sql_params)

    try:
        cursor.execute(sql, sql_params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        return {"error": f"Failed to calculate order point gaps: {err}"}

    summary = (
        f"Items ranked by gap to order point using a {recent_days}-day average usage, "
        f"seasonality from {calendar.month_name[today.month]} {seasonal_year}, "
        "and available plus on-order supply. Order points are multiplied by the seasonality factor "
        f"(capped between 0.5 and 2.0) and compared to QtyAvailable + QtyOnOrder; BuyToOrderUpTo shows the "
        "shortfall to the order-up-to level."
    )
    insights = {"summary": summary, "row_count": len(rows), "truncated": truncated}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows."

    entities = {
        "intent": "inventory",
        "metric": "order_point_gap",
        "month_context": today.month,
        "year_context": today.year,
    }
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_mrp_planning(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Advanced MRP handler for 'what should we buy' questions.
    Shows items with projected demand exceeding available supply.
    """
    lower = prompt.lower()
    buy_tokens = ("what should we buy", "what should i buy", "what to buy", "purchasing plan", "restock", "replenish", "shortage", "plan to buy", "look at", "attention", "check items", "monitor", "buy in")
    if not any(tok in lower for tok in buy_tokens):
        return None

    # Fix: Use 'future' preference so "January" in Dec 2025 means Jan 2026
    month, year = parse_month_year_from_prompt(prompt, today, preference="future", prefer_same_year=False)
    
    if not month:
        # Default to next month if not specified
        next_month_date = today + datetime.timedelta(days=30)
        month = next_month_date.month
        year = next_month_date.year
    
    # Safety Check: If parsed date is in the past, bump to next year
    # (Except if user explicitly said "last year", handled by parsing_utils logic, but here we assume planning is forward)
    # Actually, parse_month_year_from_prompt with preference='future' should handle this.
    # But let's verify:
    if year < today.year or (year == today.year and month < today.month):
         # If we somehow got a past date for a BUY signal, assume next year
         # unless explicit year was mentioned in the prompt (checked inside parsing_utils usually)
         # We'll just trust preference='future' for now.
         pass
    
    # Define the window
    start_date = today
    # End of the target month
    _, last_day = calendar.monthrange(year, month)
    end_date = datetime.date(year, month, last_day)
    
    # Lookback for historical usage (last 90 days)
    lookback_start = today - datetime.timedelta(days=90)
    primary_location = PRIMARY_LOCATION

    query = f"""
        WITH SalesDemand AS (
            SELECT 
                d.ITEMNMBR, 
                SUM(d.QUANTITY) as DemandQty
            FROM SOP10200 d
            JOIN SOP10100 h ON d.SOPNUMBE = h.SOPNUMBE
            WHERE h.REQSHIPDATE BETWEEN ? AND ?
            AND h.SOPTYPE = 2 -- Orders
            AND d.LOCNCODE = ?
            GROUP BY d.ITEMNMBR
        ),
        PurchaseSupply AS (
            SELECT 
                l.ITEMNMBR, 
                SUM(l.QTYORDER - l.QTYCANCE) as SupplyQty
            FROM POP10110 l
            WHERE l.PRMSHPDTE BETWEEN ? AND ?
            AND l.POLNESTA = 1 -- New
            AND l.LOCNCODE = ?
            GROUP BY l.ITEMNMBR
        ),
        HistoricalUsage AS (
            SELECT
                t.ITEMNMBR,
                ABS(SUM(t.TRXQTY)) as Usage90Days
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            WHERE h.DOCDATE BETWEEN ? AND ?
            AND t.TRXQTY < 0
            GROUP BY t.ITEMNMBR
        ),
        ItemStock AS (
            SELECT
                ITEMNMBR,
                SUM(QTYONHND) as TotalOnHand,
                SUM(ATYALLOC) as TotalAllocated
            FROM IV00102
            WHERE LOCNCODE = ?
            GROUP BY ITEMNMBR
        )
        SELECT TOP {CUSTOM_SQL_MAX_ROWS}
            i.ITEMNMBR,
            i.ITEMDESC,
            ISNULL(stk.TotalOnHand, 0) AS OnHand,
            ISNULL(stk.TotalAllocated, 0) AS Allocated,
            ISNULL(stk.TotalOnHand, 0) - ISNULL(stk.TotalAllocated, 0) AS CurrentFreeStock,
            ISNULL(d.DemandQty, 0) AS FirmDemand,
            ISNULL(hist.Usage90Days / 3, 0) AS MonthlyForecast,
            ISNULL(s.SupplyQty, 0) AS OnOrder,
            
            -- Projected Demand = Max of Firm or Forecast
            CASE 
                WHEN ISNULL(d.DemandQty, 0) > ISNULL(hist.Usage90Days / 3, 0) 
                THEN ISNULL(d.DemandQty, 0) 
                ELSE ISNULL(hist.Usage90Days / 3, 0) 
            END AS ProjectedDemand,
            
            -- Projected Balance = Free Stock + On Order - Projected Demand
            (ISNULL(stk.TotalOnHand, 0) - ISNULL(stk.TotalAllocated, 0)) + ISNULL(s.SupplyQty, 0) - 
            CASE WHEN ISNULL(d.DemandQty, 0) > ISNULL(hist.Usage90Days / 3, 0) 
                 THEN ISNULL(d.DemandQty, 0) 
                 ELSE ISNULL(hist.Usage90Days / 3, 0) 
            END AS ProjectedBalance,
            
            -- Shortage = How much we're short
            CASE 
                WHEN (ISNULL(stk.TotalOnHand, 0) - ISNULL(stk.TotalAllocated, 0)) + ISNULL(s.SupplyQty, 0) < 
                     CASE WHEN ISNULL(d.DemandQty, 0) > ISNULL(hist.Usage90Days / 3, 0) 
                          THEN ISNULL(d.DemandQty, 0) 
                          ELSE ISNULL(hist.Usage90Days / 3, 0) 
                     END
                THEN CASE WHEN ISNULL(d.DemandQty, 0) > ISNULL(hist.Usage90Days / 3, 0) 
                          THEN ISNULL(d.DemandQty, 0) 
                          ELSE ISNULL(hist.Usage90Days / 3, 0) 
                     END - (ISNULL(stk.TotalOnHand, 0) - ISNULL(stk.TotalAllocated, 0)) - ISNULL(s.SupplyQty, 0)
                ELSE 0 
            END AS SuggestedBuyQty,

            -- Days of Supply Code
            CASE 
                WHEN ISNULL(hist.Usage90Days, 0) = 0 THEN 999 
                ELSE (ISNULL(stk.TotalOnHand, 0) - ISNULL(stk.TotalAllocated, 0)) / (NULLIF(ISNULL(hist.Usage90Days, 0), 0) / 90.0) 
            END AS DaysOfSupply
        FROM IV00101 i
        LEFT JOIN ItemStock stk ON i.ITEMNMBR = stk.ITEMNMBR
        LEFT JOIN SalesDemand d ON d.ITEMNMBR = i.ITEMNMBR
        LEFT JOIN PurchaseSupply s ON s.ITEMNMBR = i.ITEMNMBR
        LEFT JOIN HistoricalUsage hist ON hist.ITEMNMBR = i.ITEMNMBR
        WHERE i.ITEMTYPE = 1 -- Sales Inventory
        AND (ISNULL(d.DemandQty, 0) > 0 OR ISNULL(hist.Usage90Days, 0) > 0)
        ORDER BY SuggestedBuyQty DESC, ProjectedBalance ASC
    """

    params = [
        start_date,
        end_date,
        primary_location,
        start_date,
        end_date,
        primary_location,
        lookback_start,
        today,
        primary_location,
    ]
    sql_preview = format_sql_preview(query, params)

    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        return {"error": f"Failed to generate purchasing plan: {err}"}

    # If no rows returned, return empty
    if not rows:
        return {"insights": {"summary": f"No items with demand or usage found for {calendar.month_name[month]} {year}."}}

    # Filter for actionable items (shortage > 0)
    actionable_rows = [r for r in rows if r.get("SuggestedBuyQty", 0) > 0]
    
    if actionable_rows:
        display_rows = actionable_rows
        summary = (
            f"**Purchasing Plan for {calendar.month_name[month]} {year}**  \n"
            f"Found **{len(actionable_rows)} items** with projected shortages.  \n"
            f"Demand is the greater of Firm Orders or Forecast (90-day avg ÷ 3)."
        )
    else:
        # Fallback: Show top items by demand even if no shortages
        display_rows = rows[:50]
        summary = (
            f"**No shortages projected for {calendar.month_name[month]} {year}**  \n"
            f"Showing top {len(display_rows)} items by projected demand for planning."
        )

    insights = {
        "summary": summary, 
        "row_count": len(display_rows),
        "truncated": truncated,
        "note": f"Planning Target: {month}/{year}. Window: {start_date.isoformat()} to {end_date.isoformat()}. Historical usage from last 90 days."
    }
    
    entities = {
        "intent": "planning",
        "month": month,
        "year": year,
        "target_date": end_date.isoformat()
    }

    return {"data": display_rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_what_if_analysis(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Handle what-if scenario questions involving demand changes and their impact on raw materials.
    Examples: "If SOARBLM02 demand increases 3%, what raw materials to buy?"
    """
    lower = prompt.lower()
    
    # Detect what-if keywords
    scenario_keywords = ("if", "what if", "suppose", "assume", "scenario", "hypothetical")
    if not any(k in lower for k in scenario_keywords):
        return None
    
    # Detect percentage change
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:increase|growth|grow|rise|up)', lower)
    if not pct_match:
        pct_match = re.search(r'(?:increase|growth|grow|rise|up).*?(\d+(?:\.\d+)?)\s*%', lower)
    
    if not pct_match:
        return None
    
    pct_change = float(pct_match.group(1))
    
    # Extract item if mentioned
    item = extract_item_from_prompt(prompt)
    primary_location = PRIMARY_LOCATION
    
    # Determine scope
    if item:
        scope = f"for {item}"
        item_filter = f"AND cd.ITEMNMBR = '{item}'"
    else:
        # Apply to all finished goods
        scope = "across all finished goods"
        item_filter = ""
    
    # Get BOM-based raw material impact
    query = f"""
        WITH CurrentDemand AS (
            -- Current monthly demand (last 30 days usage as baseline)
            SELECT 
                t.ITEMNMBR,
                ABS(SUM(t.TRXQTY)) as CurrentMonthlyUsage
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            WHERE h.DOCDATE >= DATEADD(day, -30, ?)
            AND t.TRXQTY < 0
            GROUP BY t.ITEMNMBR
        ),
        FinishedGoodsDemand AS (
            SELECT 
                cd.ITEMNMBR as FinishedGood,
                cd.CurrentMonthlyUsage as CurrentDemand,
                cd.CurrentMonthlyUsage * (1 + ? / 100.0) as NewDemand,
                cd.CurrentMonthlyUsage * (? / 100.0) as IncrementalDemand
            FROM CurrentDemand cd
            JOIN IV00101 i ON cd.ITEMNMBR = i.ITEMNMBR
            WHERE i.ITEMTYPE = 1 -- Finished goods
            {item_filter}
        ),
        BOMComponents AS (
            SELECT 
                b.PPN_I as FinishedGood,
                b.CPN_I as Component,
                b.QUANTITY_I as QtyPerUnit
            FROM BM010115 b
        ),
        RawMaterialImpact AS (
            SELECT 
                bc.Component,
                i.ITEMDESC as ComponentDesc,
                SUM(fg.CurrentDemand * bc.QtyPerUnit) as CurrentRequirement,
                SUM(fg.NewDemand * bc.QtyPerUnit) as NewRequirement,
                SUM(fg.IncrementalDemand * bc.QtyPerUnit) as IncrementalRequirement
            FROM BOMComponents bc
            JOIN FinishedGoodsDemand fg ON bc.FinishedGood = fg.FinishedGood
            LEFT JOIN IV00101 i ON bc.Component = i.ITEMNMBR
            GROUP BY bc.Component, i.ITEMDESC
        ),
        CurrentInventory AS (
            SELECT 
                ITEMNMBR,
                SUM(QTYONHND - ATYALLOC) as FreeStock
            FROM IV00102
            WHERE LOCNCODE = ?
            GROUP BY ITEMNMBR
        )
        SELECT TOP {CUSTOM_SQL_MAX_ROWS}
            rm.Component as ITEMNMBR,
            rm.ComponentDesc as ITEMDESC,
            ISNULL(inv.FreeStock, 0) as CurrentFreeStock,
            rm.CurrentRequirement,
            rm.NewRequirement,
            rm.IncrementalRequirement,
            rm.NewRequirement - ISNULL(inv.FreeStock, 0) as ShortageUnderScenario,
            CASE 
                WHEN rm.NewRequirement > ISNULL(inv.FreeStock, 0)
                THEN rm.NewRequirement - ISNULL(inv.FreeStock, 0)
                ELSE 0
            END as SuggestedBuyQty
        FROM RawMaterialImpact rm
        LEFT JOIN CurrentInventory inv ON rm.Component = inv.ITEMNMBR
        WHERE rm.IncrementalRequirement > 0
        ORDER BY rm.IncrementalRequirement DESC
    """
    
    params = [today, pct_change, pct_change, primary_location]
    sql_preview = format_sql_preview(query, params)
    
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        LOGGER.warning(f"What-if analysis failed: {err}")
        return None
    
    if not rows:
        insights = {
            "summary": f"No BOM components found {scope}. Cannot calculate raw material impact.",
            "row_count": 0,
            "truncated": False,
        }
        entities = {
            "intent": "what_if",
            "scenario": f"{pct_change}% increase",
            "item": item if item else "all",
            "scope": scope,
        }
        return {"data": [], "insights": insights, "sql": sql_preview, "entities": entities}
    
    # Calculate totals
    total_incremental = sum(r.get("IncrementalRequirement", 0) for r in rows)
    shortage_count = sum(1 for r in rows if r.get("SuggestedBuyQty", 0) > 0)
    
    summary = (
        f"**What-If Analysis: {pct_change}% Demand Increase {scope.title()}**\n\n"
        f"Analyzed impact on **{len(rows)} raw material components**.  \n"
        f"Found **{shortage_count} items** that would require additional purchasing.  \n"
        f"Total incremental requirement change: {total_incremental:,.0f} units."
    )
    
    insights = {
        "summary": summary,
        "row_count": len(rows),
        "truncated": truncated,
        "note": f"Scenario: {pct_change}% increase applied to monthly usage baseline (last 30 days). BOM explosion shows cascading raw material needs."
    }
    
    entities = {
        "intent": "what_if",
        "scenario": f"{pct_change}% increase",
        "item": item if item else "all",
        "scope": scope
    }
    
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_scenario_comparison(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Compare multiple what-if scenarios side-by-side.
    Example: "Compare 5% vs 10% vs 15% growth scenarios for SOARBLM02"
    """
    lower = prompt.lower()
    
    # Detect comparison keywords
    comparison_keywords = ("compare", "versus", "vs", "contrast", "difference between")
    if not any(k in lower for k in comparison_keywords):
        return None
    
    # Detect scenario/percentage keywords
    scenario_keywords = ("scenario", "growth", "increase", "%")
    if not any(k in lower for k in scenario_keywords):
        return None
    
    # Extract multiple percentages
    import re
    pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', lower)
    if len(pct_matches) < 2:
        return None  # Need at least 2 scenarios to compare
    
    percentages = [float(p) for p in pct_matches[:3]]  # Max 3 scenarios
    
    # Extract item if mentioned
    item = extract_item_from_prompt(prompt)
    primary_location = PRIMARY_LOCATION
    
    # Determine scope
    if item:
        scope = f"for {item}"
        item_filter = f"AND cd.ITEMNMBR = '{item}'"
    else:
        scope = "across all finished goods"
        item_filter = ""
    
    # Build comparison query that pivots scenarios side-by-side
    scenario_cols = []
    for i, pct in enumerate(percentages, 1):
        scenario_cols.append(
            f"cd.CurrentMonthlyUsage * (1 + {pct} / 100.0) as Scenario{i}_Demand, "
            f"cd.CurrentMonthlyUsage * ({pct} / 100.0) as Scenario{i}_Incremental"
        )
    
    scenario_select = ",\n            ".join(scenario_cols)
    
    # For each scenario, calculate raw material needs
    bom_calcs = []
    for i, pct in enumerate(percentages, 1):
        bom_calcs.append(
            f"SUM(fg.Scenario{i}_Incremental * bc.QtyPerUnit) as Scenario{i}_RawMaterialNeed"
        )
    
    bom_select = ",\n            ".join(bom_calcs)
    
    query = f"""
        WITH CurrentDemand AS (
            SELECT 
                t.ITEMNMBR,
                ABS(SUM(t.TRXQTY)) as CurrentMonthlyUsage
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            WHERE h.DOCDATE >= DATEADD(day, -30, ?)
            AND t.TRXQTY < 0
            GROUP BY t.ITEMNMBR
        ),
        FinishedGoodsScenarios AS (
            SELECT 
                cd.ITEMNMBR as FinishedGood,
                cd.CurrentMonthlyUsage as CurrentDemand,
                {scenario_select}
            FROM CurrentDemand cd
            JOIN IV00101 i ON cd.ITEMNMBR = i.ITEMNMBR
            WHERE i.ITEMTYPE = 1
            {item_filter}
        ),
        BOMComponents AS (
            SELECT 
                b.PPN_I as FinishedGood,
                b.CPN_I as Component,
                b.QUANTITY_I as QtyPerUnit
            FROM BM010115 b
        ),
        RawMaterialComparison AS (
            SELECT 
                bc.Component,
                i.ITEMDESC as ComponentDesc,
                {bom_select}
            FROM BOMComponents bc
            JOIN FinishedGoodsScenarios fg ON bc.FinishedGood = fg.FinishedGood
            LEFT JOIN IV00101 i ON bc.Component = i.ITEMNMBR
            GROUP BY bc.Component, i.ITEMDESC
        ),
        CurrentInventory AS (
            SELECT 
                ITEMNMBR,
                SUM(QTYONHND - ATYALLOC) as FreeStock
            FROM IV00102
            WHERE LOCNCODE = ?
            GROUP BY ITEMNMBR
        )
        SELECT TOP {CUSTOM_SQL_MAX_ROWS}
            rm.Component as ITEMNMBR,
            rm.ComponentDesc as ITEMDESC,
            ISNULL(inv.FreeStock, 0) as CurrentFreeStock,
            {', '.join([f'rm.Scenario{i+1}_RawMaterialNeed as Scenario{i+1}_{int(percentages[i])}pct' for i in range(len(percentages))])},
            {', '.join([f'CASE WHEN rm.Scenario{i+1}_RawMaterialNeed > ISNULL(inv.FreeStock, 0) THEN rm.Scenario{i+1}_RawMaterialNeed - ISNULL(inv.FreeStock, 0) ELSE 0 END as Shortage{i+1}_{int(percentages[i])}pct' for i in range(len(percentages))])}
        FROM RawMaterialComparison rm
        LEFT JOIN CurrentInventory inv ON rm.Component = inv.ITEMNMBR
        WHERE {' OR '.join([f'rm.Scenario{i+1}_RawMaterialNeed > 0' for i in range(len(percentages))])}
        ORDER BY {f'rm.Scenario{len(percentages)}_RawMaterialNeed DESC'}
    """
    
    params = [today, primary_location]
    sql_preview = format_sql_preview(query, params)
    
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        LOGGER.warning(f"Scenario comparison failed: {err}")
        return None
    
    if not rows:
        insights = {
            "summary": f"No BOM components found {scope} for scenario comparison.",
            "row_count": 0,
            "truncated": False,
        }
        entities = {
            "intent": "scenario_comparison",
            "scenarios": [f"{p}% growth" for p in percentages],
            "item": item if item else "all",
            "scope": scope,
        }
        return {"data": [], "insights": insights, "sql": sql_preview, "entities": entities}
    
    # Calculate summary metrics for each scenario
    scenario_summaries = []
    for i, pct in enumerate(percentages, 1):
        need_col = f"Scenario{i}_{int(pct)}pct"
        shortage_col = f"Shortage{i}_{int(pct)}pct"
        
        total_need = sum(r.get(need_col, 0) for r in rows)
        shortage_count = sum(1 for r in rows if r.get(shortage_col, 0) > 0)
        total_shortage = sum(r.get(shortage_col, 0) for r in rows)
        
        scenario_summaries.append(
            f"**{pct}% Growth**: {total_need:,.0f} total units needed, "
            f"{shortage_count} items short ({total_shortage:,.0f} units)"
        )
    
    summary = (
        f"**Scenario Comparison: {' vs '.join([f'{p}%' for p in percentages])} Growth {scope.title()}**\n\n"
        f"Comparing {len(rows)} raw material components across {len(percentages)} scenarios:\n" +
        "\n".join(scenario_summaries)
    )
    
    insights = {
        "summary": summary,
        "row_count": len(rows),
        "truncated": truncated,
        "note": f"Side-by-side comparison shows incremental raw material needs and shortages for each growth scenario. "
                f"Based on last 30 days baseline demand."
    }
    
    entities = {
        "intent": "scenario_comparison",
        "scenarios": [f"{p}% growth" for p in percentages],
        "item": item if item else "all",
        "scope": scope
    }
    
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_bom_for_item(cursor: pyodbc.Cursor, prompt: str) -> dict | None:
    """
    Handle direct manufacturing BOM inquiries like 'what is in the BOM for SOARBLM02?'
    """
    lower = prompt.lower()
    bom_tokens = (
        "bom",
        "bill of materials",
        "made of",
        "made from",
        "components",
        "what is in",
        "what are the components",
        "what is the bom",
        "what is in the bill",
        "composition",
        "recipe",
        "ingredients",
        "mfg bom",
        "manufacturing bom",
        "manufacturing bill of materials",
    )
    if not any(tok in lower for tok in bom_tokens):
        return None

    item = extract_item_from_prompt(prompt)
    if not item:
        return None

    rows, sql_preview = fetch_mfg_bom_grouped_by_component(cursor, item)

    # Fallback: try normalized '00' parent if direct item lookup fails (e.g. SOARBLM02 -> SOARBLM00)
    if not rows:
        normalized = normalize_item_for_bom(item)
        if normalized and normalized != item:
            rows, sql_preview_norm = fetch_mfg_bom_grouped_by_component(cursor, normalized)
            if rows:
                sql_preview = sql_preview_norm
                # Note: The rows will contain the Normalized Parent as PPN_I
    
    if rows:
        columns = [c[0] for c in cursor.description] if cursor.description else []
        dict_rows = [dict(zip(columns, r)) for r in rows]
        
        # Check which parent was actually found
        found_parent = dict_rows[0].get("ParentItem", item)
        parent_desc = dict_rows[0].get("ParentDescription", "")
        
        summary = f"Manufacturing Bill of Materials for {found_parent}"
        if parent_desc:
            summary += f" ({parent_desc})"
        summary += " from BM010115, grouped by component."
        
        if found_parent != item:
             summary += f" (Normalized from {item})"

        insights = {"summary": summary, "row_count": len(dict_rows)}
        entities = {"item": item, "intent": "mfg_bom"}
        if found_parent != item:
            entities["normalized_item"] = found_parent
            
        return {"data": dict_rows, "insights": insights, "sql": sql_preview, "entities": entities}

    summary = (
        f"No manufacturing BOM components were found for {item}. "
        "Check the Bill of Materials inquiry to confirm the BOM type/site."
    )
    return {
        "insights": {"summary": summary},
        "sql": sql_preview,
        "entities": {"item": item, "intent": "mfg_bom"},
    }


def handle_component_where_used(cursor: pyodbc.Cursor, prompt: str) -> dict | None:
    """
    Handle 'where-used' BOM questions such as 'what items use NPK3011?' by listing parent items
    whose BOM includes the requested component.
    """
    lower = prompt.lower()
    where_used_tokens = (
        "in them",
        "in it",
        "contains",
        "contain",
        "containing",
        "include",
        "including",
        "includes",
        "component",
        "where used",
        "use in",
        "used in",
        "uses in",
        "made with",
        "made of",
        "have a lot of",
    )
    if not any(token in lower for token in where_used_tokens):
        simple_use_pattern = "use" in lower and any(word in lower for word in ("item", "items", "product", "products", "bom"))
        if not simple_use_pattern:
            return None

    component = extract_item_from_prompt(prompt)
    if not component:
        return None

    query = """
        SELECT
            b.ITEMNMBR AS ParentItem,
            i.ITEMDESC AS ParentDescription,
            b.CMPTITNM AS ComponentItem,
            b.Design_Qty
        FROM BM00111 b
        LEFT JOIN IV00101 i ON i.ITEMNMBR = b.ITEMNMBR
        WHERE b.CMPTITNM = ?
        ORDER BY b.Design_Qty DESC, b.ITEMNMBR
    """
    try:
        cursor.execute(query, component)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        return {"error": f"Failed to query BOM usage for {component}: {err}"}

    sql_preview = format_sql_preview(query, [component])

    if not rows:
        summary = (
            f"No BOM parents were found that include component {component}. "
            "Check the Bill of Materials inquiry (Inventory > Inquiry > Bill of Materials) to verify whether the component is defined under the correct site or item number."
        )
        return {
            "insights": {"summary": summary},
            "sql": sql_preview,
            "entities": {"component": component, "intent": "bom_where_used"},
        }

    summary = (
        f"Items whose BOM includes component {component}, sorted by design quantity from BM00111. "
        "Use the Bill of Materials inquiry to validate quantities or site-specific BOMs."
    )
    insights = {"summary": summary, "row_count": len(rows), "truncated": truncated}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows."

    entities = {"component": component, "intent": "bom_where_used"}
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_item_sales_month(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """Handle 'sales of ITEM in January' style questions deterministically using shipped history."""
    sales_keywords = ("sales", "sold", "shipments", "shipped")
    if not any(k in prompt.lower() for k in sales_keywords):
        return None
    item = extract_item_from_prompt(prompt)
    if not item:
        return None
    month, year = parse_month_year_from_prompt(prompt, today, preference="past")
    if month is None or year is None:
        return None

    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"

    query = """
        SELECT SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS SalesQuantity
        FROM SOP30300 l
        JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        WHERE l.ITEMNMBR = ?
          AND h.DOCDATE BETWEEN ? AND ?
          AND l.SOPTYPE IN (3, 4)
    """
    try:
        cursor.execute(query, item, start_date, end_date)
        row = cursor.fetchone()
        sales_qty = row.SalesQuantity if row and row.SalesQuantity is not None else 0
    except pyodbc.Error as err:
        return {"error": f"Failed to query sales history: {err}"}

    sql_preview = format_sql_preview(query, [item, start_date, end_date])
    summary = (
        f"Sales for {item} in {calendar.month_name[month]} {year} based on shipped invoice history "
        f"(returns netted)."
    )
    return {
        "data": [{"SalesQuantity": sales_qty}],
        "insights": {"summary": summary, "row_count": 1},
        "sql": sql_preview,
        "entities": {"item": item, "month": month, "year": year, "intent": "sales"},
    }


from constants import RAW_MATERIAL_CATEGORIES, RAW_MATERIAL_PREFIXES


def handle_ppv_analysis(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Analyze Purchase Price Variance (PPV) for items bought recently.
    Identifies items where Unit Cost varies significantly from Standard Cost.
    """
    lower = prompt.lower()
    ppv_tokens = ("ppv", "price variance", "purchase price variance", "cost variance", "standard cost variance", "bad pricing")
    if not any(tok in lower for tok in ppv_tokens):
        return None

    # Default to last 30 days unless specified
    months_mentioned = _extract_months_from_prompt(prompt)
    if "last month" in lower or "past month" in lower or (months_mentioned and len(months_mentioned) == 1):
        # Simplification: just grab last 30 days for now to keep it robust
        start_date = today - datetime.timedelta(days=30)
    elif "last 3 months" in lower or "past quarter" in lower:
        start_date = today - datetime.timedelta(days=90)
    else:
        start_date = today - datetime.timedelta(days=30)

    rm_category_list = "', '".join(RAW_MATERIAL_CATEGORIES)
    rm_prefix_conditions = " OR ".join([f"i.ITEMNMBR LIKE '{p}%'" for p in RAW_MATERIAL_PREFIXES])

    # Query: Join POP30110 (Line History) with IV00101 (Item Master for Standard Cost)
    # Filter: Only items with Standard Cost > 0 to avoid div/0 AND raw materials only
    query = f"""
        SELECT TOP {CUSTOM_SQL_MAX_ROWS}
            l.ITEMNMBR,
            i.ITEMDESC,
            AVG(l.UNITCOST) as AvgUnitCost,
            MAX(i.STNDCOST) as StandardCost,
            SUM(l.QTYINVCD) as QtyInvoiced,
            SUM((l.UNITCOST - i.STNDCOST) * l.QTYINVCD) as TotalPPV,
            AVG(CASE WHEN i.STNDCOST > 0 THEN ((l.UNITCOST - i.STNDCOST) / i.STNDCOST) * 100 ELSE 0 END) as AvgVariancePct
        FROM POP30110 l
        JOIN POP30100 h ON l.PONUMBER = h.PONUMBER
        LEFT JOIN IV00101 i ON l.ITEMNMBR = i.ITEMNMBR
        WHERE h.RECEIPTDATE >= ?
          AND i.STNDCOST > 0
          AND (
                i.ITMCLSCD IN ('{rm_category_list}') 
                OR {rm_prefix_conditions}
          )
        GROUP BY l.ITEMNMBR, i.ITEMDESC
        HAVING ABS(SUM((l.UNITCOST - i.STNDCOST) * l.QTYINVCD)) > 100 -- Filter for material variance > $100
        ORDER BY TotalPPV DESC
    """
    
    params = [start_date]
    sql_preview = format_sql_preview(query, params)
    
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        return {"error": f"Failed to calculate PPV: {err}"}
        
    if not rows:
        return {
            "insights": {"summary": "No significant purchase variations found (> $100 total variance) in the selected period."},
            "sql": sql_preview
        }

    # Summarize
    total_unfav = sum(r["TotalPPV"] for r in rows if r["TotalPPV"] > 0)
    total_fav = sum(r["TotalPPV"] for r in rows if r["TotalPPV"] < 0)
    net_ppv = total_unfav + total_fav
    
    summary = (
        f"**Purchase Price Variance Analysis ({start_date} to {today})**\n\n"
        f"- **Unfavorable Variance:** ${total_unfav:,.2f}\n"
        f"- **Favorable Variance:** ${total_fav:,.2f}\n"
        f"- **Net Impact:** ${net_ppv:,.2f}\n\n"
        f"Showing top items driving price variance against Standard Cost."
    )
    
    insights = {
        "summary": summary,
        "row_count": len(rows),
        "truncated": truncated
    }
    
    entities = {"intent": "ppv_analysis", "net_variance": net_ppv}
    
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_forecast_bom_requirements(
    cursor: pyodbc.Cursor, prompt: str, today: datetime.date
) -> dict | None:
    """
    Handle questions like "If we expect GOLDCA02 sales to increase by 8% in January, what raw materials do we need?"
    Uses shipped history or open orders for the specified month, applies growth, explodes BOM (with 00-normalization),
    nets against on-hand and open PO supply, and returns requirements and incremental purchases.
    """
    pct = parse_percent_increase(prompt)
    if pct is None:
        return None

    item = extract_item_from_prompt(prompt)
    if not item:
        return None

    month, year = parse_month_year_from_prompt(prompt, today, preference="future", prefer_same_year=True)
    if month is None or year is None:
        return None

    factor = Decimal("1") + Decimal(str(pct))

    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
    requested_year = year
    target_period_start = datetime.date(year, month, 1)
    is_future_period = target_period_start > today

    shipped_query = """
        SELECT SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS ShippedQty
        FROM SOP30300 l
        JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        WHERE l.ITEMNMBR = ?
          AND h.DOCDATE BETWEEN ? AND ?
          AND l.SOPTYPE IN (3, 4)
    """
    demand_sql_used = shipped_query
    demand_params = [item, start_date, end_date]
    try:
        cursor.execute(shipped_query, item, start_date, end_date)
        shipped_qty_row = cursor.fetchone()
        shipped_qty = decimal_or_zero(shipped_qty_row.ShippedQty) if shipped_qty_row else Decimal("0")
    except pyodbc.Error as err:
        return {"error": f"Failed to query shipped history: {err}"}

    base_source = "shipped invoice history"
    open_query = None
    if shipped_qty == 0:
        open_query = """
            SELECT SUM(l.QUANTITY) AS OpenQty
            FROM SOP10200 l
            JOIN SOP10100 h ON l.SOPNUMBE = h.SOPNUMBE
            WHERE l.ITEMNMBR = ?
              AND h.DOCDATE BETWEEN ? AND ?
        """
        try:
            cursor.execute(open_query, item, start_date, end_date)
            open_row = cursor.fetchone()
            shipped_qty = decimal_or_zero(open_row.OpenQty) if open_row else Decimal("0")
            base_source = "open sales orders"
            demand_sql_used = open_query
            demand_params = [item, start_date, end_date]
        except pyodbc.Error as err:
            return {"error": f"Failed to query open sales orders: {err}"}

    initial_period_sql = format_sql_preview(demand_sql_used, demand_params)
    fallback_note = None
    fallback_sql = None
    if shipped_qty == 0 and is_future_period:
        fallback_year = requested_year - 1
        fallback_start = f"{fallback_year}-{month:02d}-01"
        fallback_end = f"{fallback_year}-{month:02d}-{calendar.monthrange(fallback_year, month)[1]}"
        try:
            cursor.execute(shipped_query, item, fallback_start, fallback_end)
            fallback_row = cursor.fetchone()
            fallback_qty = decimal_or_zero(fallback_row.ShippedQty) if fallback_row else Decimal("0")
            fallback_sql = format_sql_preview(shipped_query, [item, fallback_start, fallback_end])
        except pyodbc.Error as err:
            return {"error": f"Failed to query shipped history fallback: {err}"}

        if fallback_qty > 0:
            shipped_qty = fallback_qty
            year = fallback_year
            start_date = fallback_start
            end_date = fallback_end
            base_source = (
                f"shipped invoice history from {calendar.month_name[month]} {fallback_year} "
                f"(fallback because {calendar.month_name[month]} {requested_year} has no demand yet)"
            )
            demand_sql_used = shipped_query
            demand_params = [item, start_date, end_date]
        else:
            fallback_note = (
                f"No demand found for future period {calendar.month_name[month]} {requested_year}; "
                f"also checked {calendar.month_name[month]} {fallback_year} history."
            )

    if shipped_qty == 0:
        return {"insights": {"summary": f"No shipped or open-order demand found for {item} in {calendar.month_name[month]} {year}."}}

    expected_qty = shipped_qty * factor

    normalized_parent = normalize_item_for_bom(item)
    bom_candidates = []
    if normalized_parent:
        bom_candidates.append(normalized_parent)
    if item not in bom_candidates:
        bom_candidates.append(item)

    bom_components = []
    bom_sql = ""
    bom_parent_used = None
    for candidate in bom_candidates:
        comps, bom_sql_candidate = fetch_recursive_bom_for_item(cursor, candidate)
        if bom_sql == "":
            bom_sql = bom_sql_candidate
        if comps:
            bom_components = comps
            bom_sql = bom_sql_candidate
            bom_parent_used = candidate
            break
    normalization_note = None
    if bom_parent_used and bom_parent_used != item:
        normalization_note = f"Normalized {item} to {bom_parent_used} (00 parent) for BOM explosion."

    if not bom_components:
        return {"insights": {"summary": f"No BOM rows found for {item}. Check the BOM screen to verify the parent item or site."}}

    results = []
    for comp in bom_components:
        design_qty = decimal_or_zero(comp.Design_Qty)
        base_requirement = design_qty * shipped_qty
        expected_requirement = design_qty * expected_qty
        results.append({
            "RawMaterial": comp.RawMaterial,
            "DesignQtyPerFG": design_qty,
            "BaseDemand": base_requirement,
            "ExpectedDemand": expected_requirement,
            "Incremental": expected_requirement - base_requirement,
            "SourceBOMParent": bom_parent_used or item,
        })

    rm_items = [row["RawMaterial"] for row in results]
    on_hand, on_hand_sql = fetch_on_hand_by_item(cursor, rm_items)
    open_po, open_po_sql = fetch_open_po_supply(cursor, rm_items)

    for row in results:
        rm = row["RawMaterial"]
        on_hand_qty = on_hand.get(rm, Decimal("0"))
        open_po_qty = open_po.get(rm, Decimal("0"))

        base_net = max(row["BaseDemand"] - on_hand_qty - open_po_qty, Decimal("0"))
        net_expected = max(row["ExpectedDemand"] - on_hand_qty - open_po_qty, Decimal("0"))
        incremental_purchase = max(net_expected - base_net, Decimal("0"))

        row.update({
            "OnHand": on_hand_qty,
            "OpenPO": open_po_qty,
            "BaseNetRequirement": base_net,
            "NetExpectedRequirement": net_expected,
            "IncrementalPurchaseNeed": incremental_purchase,
        })

    results.sort(key=lambda x: x["NetExpectedRequirement"], reverse=True)

    summary = (
        f"Applied a {pct*100:.1f}% increase to {item} demand for {calendar.month_name[month]} {year} "
        f"using {base_source}. BOM for {bom_parent_used or item} (00-normalized when applicable) exploded to raw material requirements, "
        f"netted against on-hand (IV00102) and open purchase orders (POP10110)."
    )

    sql_bits = {
        "demand_sql": format_sql_preview(demand_sql_used, demand_params),
        "bom_sql": bom_sql,
    }
    if fallback_sql:
        sql_bits["fallback_demand_sql"] = fallback_sql
        sql_bits["initial_period_sql"] = initial_period_sql
    if on_hand_sql:
        sql_bits["on_hand_sql"] = on_hand_sql
    if open_po_sql:
        sql_bits["open_po_sql"] = open_po_sql

    insights = {"summary": summary, "row_count": len(results)}
    note_parts = []
    if normalization_note:
        note_parts.append(normalization_note)
    if fallback_note:
        note_parts.append(fallback_note)
    if note_parts:
        insights["note"] = " ".join(note_parts)
    return {"data": results, "insights": insights, "sql": sql_bits, "entities": {"item": item, "month": month, "year": year, "intent": "planning"}}


def handle_forecast_playbook(prompt: str) -> dict | None:
    """
    Provide a deterministic playbook for forecast-to-raw-material projections when the user is asking
    how to structure the logic rather than requesting a live calculation.
    """
    prompt_lower = prompt.lower()
    has_forecast = any(tok in prompt_lower for tok in ("forecast", "projection", "predict", "predicted", "schedule", "production schedule", "finished goods", "fg forecast"))
    has_material_focus = any(tok in prompt_lower for tok in ("raw material", "raw materials", "bom", "bill of materials", "blm", "kanban", "00 code", "00 codes", "component", "ingredient"))
    asking_for_plan = any(tok in prompt_lower for tok in ("plan", "approach", "logic", "strategy", "best", "how", "add this", "train"))
    if not (has_forecast and has_material_focus and asking_for_plan):
        return None

    steps = [
        {
            "Step": 1,
            "Focus": "Master BOM + density",
            "Details": (
                "One row per SKU per raw material with BOM qty per 100 lb (or batch), finished-good density, "
                "and computed RM lbs/gal = (BOM lbs per 100 lbs ÷ 100) × FG density. Normalize parents to 00 codes."
            ),
        },
        {
            "Step": 2,
            "Focus": "Forecast input",
            "Details": (
                "Store finished-good forecast by month/quarter in gallons (convert totes/drums). "
                "Allow growth factors by rep/segment and flat/down adjustments for declining SKUs."
            ),
        },
        {
            "Step": 3,
            "Focus": "Demand math",
            "Details": "Raw material demand (lbs) = Forecasted gallons × RM lbs/gal from the BOM table. Aggregate by month/quarter.",
        },
        {
            "Step": 4,
            "Focus": "Source blending",
            "Details": (
                "Near-term: use production schedule/open sales orders as the demand driver. "
                "Mid/long-term: use the forecast table. Both map finished goods to BOM parents (00-normalized) before exploding."
            ),
        },
        {
            "Step": 5,
            "Focus": "Netting and buys",
            "Details": (
                "Net gross RM demand against on-hand (IV00102) and open POs (POP10110) to get net purchase needs. "
                "Surface top gaps for procurement."
            ),
        },
        {
            "Step": 6,
            "Focus": "Outputs",
            "Details": (
                "Monthly/quarterly RM demand by material and product line, change vs prior year, and buy recommendations. "
                "Use the same table to size Kanban buffers."
            ),
        },
    ]

    summary = (
        "Forecast-to-RM playbook: keep a BOM/density bridge table (00-normalized parents), store finished-good forecasts in gallons, "
        "multiply gallons by RM lbs/gal to get raw-material demand, and blend production schedule for near-term coverage. "
        "Net the results against on-hand and open POs to drive buys and Kanban sizing."
    )
    insights = {
        "summary": summary,
        "narrative": summary,
        "row_count": len(steps),
        "note": "Use manufacturing BOM first (BM010115), standard BOM fallback (BM00111), and convert totes/drums to gallons for consistency.",
    }
    return {"data": steps, "insights": insights, "sql": None, "entities": {"intent": "planning", "topic": "forecast_playbook"}}


def handle_lot_traceability(cursor: pyodbc.Cursor, prompt: str) -> dict | None:
    """Trace a lot number through inventory transactions (IV30300)."""
    lower = prompt.lower()
    trace_keywords = ("trace", "traceability", "where did lot", "history of lot", "lot history")
    lot_keywords = ("lot", "lot number")
    if not any(k in lower for k in trace_keywords) or not any(k in lower for k in lot_keywords):
        return None

    lot_number = extract_lot_from_prompt(prompt)
    if not lot_number:
        return None

    query = """
        SELECT
            t.ITEMNMBR,
            i.ITEMDESC,
            h.DOCDATE,
            t.DOCNUMBR,
            t.DOCTYPE,
            t.TRXLOCTN,
            t.TRXQTY,
            t.TRXSORCE
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        LEFT JOIN IV00101 i ON i.ITEMNMBR = t.ITEMNMBR
        WHERE t.LOTNUMBR = ?
        ORDER BY h.DOCDATE, t.DEX_ROW_TS
    """
    sql_preview = format_sql_preview(query, [lot_number])

    try:
        cursor.execute(query, lot_number)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        return {"error": f"Failed to trace lot number {lot_number}: {err}"}

    if not rows:
        summary = f"No transaction history found for lot number {lot_number} in IV30300."
        return {"insights": {"summary": summary}, "sql": sql_preview, "entities": {"lot_number": lot_number}}

    summary = f"Transaction history for lot number {lot_number}, ordered by document date."
    insights = {"summary": summary, "row_count": len(rows), "truncated": truncated}
    if truncated:
        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} transactions."

    entities = {"lot_number": lot_number, "intent": "lot_traceability"}
    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}


def handle_mrp_style_question(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict:
    """
    Deterministic MRP-style handler: derives finished-good demand from open sales orders for the requested
    month/year (or the next logical future period), explodes BOM parents, nets against on-hand and open POs,
    and returns net raw-material purchase requirements.
    """
    def _fmt_qty(val) -> str:
        dec = decimal_or_zero(val)
        formatted = f"{dec:.4f}".rstrip("0").rstrip(".")
        return formatted or "0"

    month, year = parse_month_year_from_prompt(prompt, today, preference="future")
    if month is None or year is None:
        month, year = today.month, today.year

    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
    target_period_start = datetime.date(year, month, 1)
    is_future_period = target_period_start > today

    demand_query = """
        SELECT l.ITEMNMBR, SUM(l.QUANTITY) AS DemandQty
        FROM SOP10200 l
        JOIN SOP10100 h ON l.SOPNUMBE = h.SOPNUMBE
        WHERE h.DOCDATE BETWEEN ? AND ?
        GROUP BY l.ITEMNMBR
    """
    sql_bits: dict[str, str] = {}
    op_gap_error = None
    op_gap_note = None
    used_historical_fallback = False
    try:
        cursor.execute(demand_query, start_date, end_date)
        demand_rows = cursor.fetchall()
        sql_bits["demand_sql"] = format_sql_preview(demand_query, [start_date, end_date])
    except pyodbc.Error as err:
        return {"error": f"Failed to query open sales orders for planning: {err}"}

    demand_source = f"open sales orders in {calendar.month_name[month]} {year}"

    if not demand_rows and is_future_period:
        op_gap = handle_order_point_gap(cursor, prompt, today)
        if isinstance(op_gap, dict):
            if op_gap.get("error"):
                op_gap_error = op_gap.get("error")
            elif op_gap.get("data"):
                op_data = op_gap.get("data", [])
                op_insights = op_gap.get("insights", {}) if isinstance(op_gap.get("insights"), dict) else {}
                op_sql = op_gap.get("sql")
                if op_sql:
                    sql_bits["order_point_sql"] = op_sql

                positive = []
                for row in op_data:
                    if isinstance(row, dict):
                        gap_val = decimal_or_zero(row.get("GapToOrderPoint"))
                        buy_to = decimal_or_zero(row.get("BuyToOrderUpTo", 0))
                    else:
                        gap_val = decimal_or_zero(getattr(row, "GapToOrderPoint", 0))
                        buy_to = decimal_or_zero(getattr(row, "BuyToOrderUpTo", 0))
                    if gap_val > 0 or buy_to > 0:
                        positive.append(row)

                narrative = (
                    f"Forward plan for {calendar.month_name[month]} {year}: no open sales orders yet, so used order-point gaps "
                    f"(90-day usage plus seasonality) to surface suggested buys. "
                    f"{len(positive)} item(s) show a gap or buy-to need; on-hand and open POs are included."
                )
                insights = {
                    "summary": narrative,
                    "narrative": narrative,
                    "row_count": len(op_data),
                    "demand_signal": "order point gaps",
                    "net_purchase_count": len(positive),
                }
                if note := op_insights.get("note"):
                    insights["note"] = note

                return {
                    "data": op_data,
                    "insights": insights,
                    "sql": sql_bits,
                    "entities": {"month": month, "year": year, "intent": "planning"},
                }
            else:
                op_insights = op_gap.get("insights", {}) if isinstance(op_gap, dict) else {}
                op_gap_note = op_insights.get("note") or op_insights.get("summary")

    if not demand_rows:
        hist_start = f"{year - 1}-{month:02d}-01"
        hist_end = f"{year - 1}-{month:02d}-{calendar.monthrange(year - 1, month)[1]}"
        fallback_query = """
            SELECT l.ITEMNMBR, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS DemandQty
            FROM SOP30300 l
            JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
            WHERE h.DOCDATE BETWEEN ? AND ?
              AND l.SOPTYPE IN (3, 4)
            GROUP BY l.ITEMNMBR
        """
        try:
            cursor.execute(fallback_query, hist_start, hist_end)
            demand_rows = cursor.fetchall()
            used_historical_fallback = bool(demand_rows)
            if is_future_period:
                demand_source = (
                    f"shipped invoices in {calendar.month_name[month]} {year - 1} "
                    f"(fallback because {calendar.month_name[month]} {year} has no open orders yet)"
                )
            else:
                demand_source = (
                    f"shipped invoices in {calendar.month_name[month]} {year - 1} because no open sales orders "
                    f"were found for {calendar.month_name[month]} {year}"
                )
            sql_bits["fallback_demand_sql"] = format_sql_preview(fallback_query, [hist_start, hist_end])
        except pyodbc.Error as err:
            return {"error": f"Failed to query historical shipments for planning fallback: {err}"}

    if not demand_rows:
        summary_parts = [f"No demand signals found for {calendar.month_name[month]} {year}."]
        if is_future_period:
            summary_parts.append(
                "Checked open sales orders, order-point gaps, and last year's shipments. Provide a sales forecast to calculate purchases."
            )
        else:
            summary_parts.append("Checked open sales orders and last year's shipments.")

        insights = {
            "summary": " ".join(summary_parts),
            "row_count": 0,
            "net_purchase_count": 0,
            "demand_signal": "open orders, order-point gaps, and prior-year shipments" if is_future_period else "open orders and prior-year shipments",
        }
        note_parts = []
        if op_gap_error:
            note_parts.append(f"Order-point gaps unavailable: {op_gap_error}")
        elif op_gap_note:
            note_parts.append(op_gap_note)
        if note_parts:
            insights["note"] = " ".join(note_parts)

        return {
            "data": [],
            "insights": insights,
            "sql": sql_bits,
            "entities": {"month": month, "year": year, "intent": "planning"},
        }

    parent_demand: dict[str, Decimal] = {}
    for row in demand_rows:
        fg = getattr(row, "ITEMNMBR", None)
        qty = decimal_or_zero(getattr(row, "DemandQty", None))
        if not fg or qty == 0:
            continue
        parent = normalize_item_for_bom(fg)
        parent_demand[parent] = parent_demand.get(parent, Decimal("0")) + qty

    if not parent_demand:
        return {"insights": {"summary": "Demand rows were returned but no finished goods could be mapped to BOM parents."}}

    bom_parents = list(parent_demand.keys())
    placeholders = ", ".join("?" for _ in bom_parents)

    bom_components: list = []
    mfg_components: list = []
    mfg_parents: set[str] = set()
    fallback_parents: set[str] = set()

    mfg_query = f"""
        SELECT PPN_I as BOMParent, CPN_I as RawMaterial, SUM(QUANTITY_I) as Design_Qty
        FROM BM010115
        WHERE PPN_I IN ({placeholders})
        GROUP BY PPN_I, CPN_I
    """
    try:
        cursor.execute(mfg_query, bom_parents)
        mfg_components = cursor.fetchall()
        sql_bits["mfg_bom_sql"] = format_sql_preview(mfg_query, bom_parents)
        bom_components.extend(mfg_components)
        mfg_parents = {getattr(row, "BOMParent", None) for row in mfg_components if getattr(row, "BOMParent", None)}
    except pyodbc.Error as err:
        return {"error": f"Failed to query Manufacturing BOM (BM010115): {err}"}

    missing_parents = [p for p in bom_parents if p not in mfg_parents]
    if missing_parents:
        fallback_placeholders = ", ".join("?" for _ in missing_parents)
        bom_query = f"""
            SELECT ITEMNMBR as BOMParent, CMPTITNM as RawMaterial, Design_Qty
            FROM BM00111
            WHERE ITEMNMBR IN ({fallback_placeholders})
        """
        try:
            cursor.execute(bom_query, missing_parents)
            bom_components.extend(cursor.fetchall())
            sql_bits["bom_sql"] = format_sql_preview(bom_query, missing_parents)
            fallback_parents = set(missing_parents)
        except pyodbc.Error as err:
            return {"error": f"Failed to query Bill of Materials fallback: {err}"}

    raw_material_demand: dict[str, Decimal] = {}
    bom_source_by_rm: dict[str, str] = {}
    for comp in bom_components:
        parent_qty = parent_demand.get(comp.BOMParent, Decimal("0"))
        demand = decimal_or_zero(comp.Design_Qty) * parent_qty
        raw_material_demand[comp.RawMaterial] = raw_material_demand.get(comp.RawMaterial, Decimal("0")) + demand
        if comp.RawMaterial not in bom_source_by_rm:
            source_label = "MFG BOM" if comp.BOMParent in mfg_parents else "BOM"
            bom_source_by_rm[comp.RawMaterial] = source_label

    if not raw_material_demand:
        return {"insights": {"summary": "Found demand for finished goods, but no BOM rows to explode into raw materials."}}

    rm_items = list(raw_material_demand.keys())
    on_hand, on_hand_sql = fetch_on_hand_by_item(cursor, rm_items)
    open_po, open_po_sql = fetch_open_po_supply(cursor, rm_items)
    if on_hand_sql:
        sql_bits["on_hand_sql"] = on_hand_sql
    if open_po_sql:
        sql_bits["open_po_sql"] = open_po_sql

    results = []
    for rm_item, gross in raw_material_demand.items():
        on_hand_qty = on_hand.get(rm_item, Decimal("0"))
        open_po_qty = open_po.get(rm_item, Decimal("0"))
        net_requirement = max(gross - on_hand_qty - open_po_qty, Decimal("0"))
        results.append({
            "RawMaterial": rm_item,
            "GrossRequirement": gross,
            "OnHand": on_hand_qty,
            "OpenPO": open_po_qty,
            "NetRequirement": net_requirement,
            "BOMSource": bom_source_by_rm.get(rm_item, ""),
        })

    results.sort(key=lambda x: x["NetRequirement"], reverse=True)

    positive = [r for r in results if r["NetRequirement"] > 0]
    top_recs = positive[:3]
    if top_recs:
        top_lines = "; ".join(
            f"**{row['RawMaterial']}** needs **{_fmt_qty(row['NetRequirement'])}** "
            f"(gross {_fmt_qty(row['GrossRequirement'])}, on-hand {_fmt_qty(row['OnHand'])}, open PO {_fmt_qty(row['OpenPO'])})"
            for row in top_recs
        )
        purchase_text = f"Top buys: {top_lines}."
        if len(positive) > len(top_recs):
            purchase_text += f" {len(positive) - len(top_recs)} more material(s) also show net demand."
    else:
        purchase_text = "**No net raw-material purchases are required; on-hand and open POs cover projected demand.**"

    narrative = (
        f"For {calendar.month_name[month]} {year}, demand from **{demand_source}** was exploded through BOM production data "
        f"(BM010115) with standard BOM fallback ({len(mfg_parents)} manufacturing parents, {len(fallback_parents)} fallback) "
        f"and netted against on-hand and open PO supply. {purchase_text}"
    )
    summary = narrative
    insights = {
        "summary": summary,
        "narrative": narrative,
        "row_count": len(results),
        "demand_signal": demand_source,
        "net_purchase_count": len(positive),
    }
    note_parts = []
    if used_historical_fallback:
        note_parts.append("Used last year's shipments because the target period has no open orders.")
    if op_gap_error:
        note_parts.append(f"Order-point gaps unavailable: {op_gap_error}")
    elif op_gap_note:
        note_parts.append(op_gap_note)
    if note_parts:
        existing_note = insights.get("note")
        joined = " ".join(note_parts)
        insights["note"] = f"{existing_note} {joined}".strip() if existing_note else joined
    return {"data": results, "insights": insights, "sql": sql_bits, "entities": {"month": month, "year": year, "intent": "planning"}}


def handle_custom_sql_question(
    cursor: pyodbc.Cursor, prompt: str, today: datetime.date, context: dict | None = None, history_hint: str | None = None
) -> dict:
    context = context or {}
    llm_usage: list[dict] = []

    def _with_usage(payload: dict) -> dict:
        if llm_usage:
            payload["usage"] = llm_usage
        return payload

    # Deterministic raw-material usage: totals across months, then per-month ranking, then overall lookbacks.
    multi_month_total = handle_raw_material_usage_multi_month_total(cursor, prompt, today)
    if multi_month_total is not None:
        return multi_month_total

    # Vendor Scorecard
    vendor_scorecard = handle_vendor_scorecard(cursor, prompt, today)
    if vendor_scorecard is not None:
        return vendor_scorecard

    # PPV Analysis
    ppv_result = handle_ppv_analysis(cursor, prompt, today)
    if ppv_result is not None:
        return ppv_result

    # Deterministic raw-material usage: multi-month, single-month, and overall lookbacks.
    multi_month_usage = handle_raw_material_usage_multi_month(cursor, prompt, today)
    if multi_month_usage is not None:
        return multi_month_usage

    monthly_usage = handle_raw_material_usage_month(cursor, prompt, today)
    if monthly_usage is not None:
        return monthly_usage

    overall_usage = handle_raw_material_usage_overall(cursor, prompt, today)
    if overall_usage is not None:
        return overall_usage

    if USE_DYNAMIC_HANDLERS:
        dynamic_handler = find_dynamic_handler(prompt)
        if dynamic_handler:
            sql_text = dynamic_handler.get("sql")
            if sql_text:
                params = normalize_sql_params(dynamic_handler.get("params"))
                sql_preview = format_sql_preview(sql_text, params)
                handler_name = dynamic_handler.get("name") or dynamic_handler.get("prompt") or "Learned query"
                chart_spec = dynamic_handler.get("chart") if isinstance(dynamic_handler.get("chart"), dict) else None
                try:
                    cursor.execute(sql_text, params)
                    fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
                    columns = [col[0] for col in cursor.description] if cursor.description else []

                    rows = [dict(zip(columns, row)) for row in fetched[:CUSTOM_SQL_MAX_ROWS]]
                    truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS

                    summary_text = f"Executed dynamic handler for: {dynamic_handler.get('name') or 'previous successful query'}"

                    insights = {
                        "summary": summary_text,
                        "row_count": len(rows),
                        "truncated": truncated,
                        "note": "Used a dynamically learned handler.",
                        "handler_name": handler_name,
                    }
                    if truncated:
                        insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows. Used a dynamically learned handler."

                    return {
                        "data": rows,
                        "sql": sql_preview,
                        "raw_sql": sql_text,
                        "params": params,
                        "chart": chart_spec,
                        "insights": insights,
                        "entities": dynamic_handler.get("entities", {}),
                        "handler_name": handler_name,
                    }
                except pyodbc.Error as err:
                    LOGGER.warning("Dynamic handler '%s' failed: %s", dynamic_handler.get("name") or "<unnamed>", err)

    forecast_playbook = handle_forecast_playbook(prompt)
    if forecast_playbook is not None:
        return forecast_playbook

    forecast_result = handle_forecast_bom_requirements(cursor, prompt, today)
    if forecast_result is not None:
        return forecast_result

    # Advanced MRP / "What should I buy" handler
    mrp_result = handle_mrp_planning(cursor, prompt, today)
    if mrp_result is not None:
        return mrp_result

    # Direct item-month sales question.
    item_sales_result = handle_item_sales_month(cursor, prompt, today)
    if item_sales_result is not None:
        return item_sales_result

    # Check for top-selling items question with explicit month/year.
    top_result = handle_top_selling_question(cursor, prompt, today)
    if top_result is not None:
        return top_result

    order_point_result = handle_order_point_gap(cursor, prompt, today)
    if order_point_result is not None:
        return order_point_result
    
    # BOM components for an item
    bom_result = handle_bom_for_item(cursor, prompt)
    if bom_result is not None:
        return bom_result

    # Where-used component question (e.g., "what items use NPK3011?")
    where_used_result = handle_component_where_used(cursor, prompt)
    if where_used_result is not None:
        return where_used_result

    # Lot traceability question
    lot_traceability_result = handle_lot_traceability(cursor, prompt)
    if lot_traceability_result is not None:
        return lot_traceability_result

    # Scenario Comparison (compare multiple what-if scenarios side-by-side)
    scenario_comparison_result = handle_scenario_comparison(cursor, prompt, today)
    if scenario_comparison_result is not None:
        return scenario_comparison_result

    # What-If Analysis (single scenario-based questions)
    what_if_result = handle_what_if_analysis(cursor, prompt, today)
    if what_if_result is not None:
        return what_if_result

    # Advanced MRP Planning
    mrp_planning_result = handle_mrp_planning(cursor, prompt, today)
    if mrp_planning_result is not None:
        return mrp_planning_result

    # Check for MRP-style question
    mrp_keywords = (
        "what materials should we buy",
        "what inventory should we buy",
        "what should i buy",
        "what should we buy",
        "what to buy",
        "purchase for",
        "buy in",
        "buy for",
        "purchase in",
        "raw materials should i buy",
        "raw materials should we buy",
        "materials to buy",
        "plan to purchase",
        "what should we purchase for inventory",
        "purchase plan",
        "mrp",
        "material requirement",
        "materials resource planning",
        "procurement plan",
        "stock up",
        "replenish plan",
        "bom production",
        "production bom",
    )
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in mrp_keywords):
        return handle_mrp_style_question(cursor, prompt, today)

    schema = load_allowed_sql_schema(cursor)
    if not schema:
        return {"error": "Custom SQL requires schema metadata, but it is unavailable."}

    schema_summary = summarize_schema_for_prompt(schema, priority_columns=SCHEMA_PRIORITY_COLUMNS)
    context_hint = summarize_sql_context(context)

    date_hint = None
    pairs = _extract_month_year_pairs(prompt, today)
    if pairs:
        date_hint = "Date context from user prompt: " + ", ".join(f"{calendar.month_name[m]} {y}" for y, m in pairs)

    retry_reason = None
    previous_sql = None
    previous_params = None
    max_retries = 2
    for attempt in range(max_retries):
        if attempt > 0:
            LOGGER.info(f"Retrying SQL generation. Attempt {attempt + 1} of {max_retries}")

        plan = call_openai_sql_generator(
            prompt,
            today,
            schema_summary,
            context_hint=context_hint,
            retry_reason=retry_reason,
            conversation_hint=history_hint,
            previous_sql=previous_sql,
            previous_params=previous_params,
            date_hint=date_hint,
        )
        plan_usage = plan.get("usage") if isinstance(plan, dict) else None
        if isinstance(plan_usage, dict):
            llm_usage.append({"label": "sql_generator", **plan_usage})
        if not isinstance(plan, dict):
            return _with_usage({"error": "I wasn't able to design a custom SQL query for that request."})

        chart_spec = plan.get("chart") if isinstance(plan.get("chart"), dict) else None
        sql_text = plan.get("sql")
        valid, reason = validate_custom_sql(sql_text, CUSTOM_SQL_ALLOWED_TABLES)
        if not valid:
            if attempt < max_retries - 1:
                retry_reason = reason or "The generated SQL was not valid."
                continue
            return _with_usage({"error": reason or "The generated SQL was not valid.", "sql": sql_text})

        params = normalize_sql_params(plan.get("params"))
        previous_sql = sql_text
        previous_params = params
        sql_preview = format_sql_preview(sql_text, params)

        handler_name = plan.get("summary") or prompt or "Custom SQL Query"
        try:
            cursor.execute(sql_text, params)
            fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
            columns = [col[0] for col in cursor.description] if cursor.description else []
            
            rows = [dict(zip(columns, row)) for row in fetched[:CUSTOM_SQL_MAX_ROWS]]
            truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
            
            summary_text = plan.get("summary", "")
            bom_note = build_bom_guidance(prompt, plan, rows)
            if bom_note:
                summary_text = f"{summary_text}\n\n{bom_note}" if summary_text else bom_note

            insights = {"summary": summary_text, "row_count": len(rows), "truncated": truncated}
            if truncated:
                insights["note"] = f"Showing first {CUSTOM_SQL_MAX_ROWS} rows."
            insights["handler_name"] = handler_name

            should_learn = should_log_training_example(prompt)
            if should_learn:
                handler_data = {
                    "name": handler_name,
                    "sql": sql_text,
                    "params": params,
                    "entities": plan.get("entities", {}),
                    "prompt": prompt,
                }
                save_dynamic_handler(handler_name, handler_data)
                record_sql_example(
                    prompt=prompt,
                    summary=summary_text or plan.get("summary"),
                    sql=sql_text,
                    params=params,
                    entities=plan.get("entities", {}),
                    route="erp_sql",
                    source="llm_sql_generator",
                )
            else:
                LOGGER.debug("Skipping training capture for context-dependent prompt: %s", prompt)

            if "report_structure" in plan:
                return _with_usage({
                    "action": "navigate",
                    "page": "generative_insights",
                    "data": {
                        "structure": plan["report_structure"],
                        "data": rows
                    },
                    "insights": {"summary": summary_text or "Generative report created."}
                })

            return _with_usage({
                "data": rows,
                "sql": sql_preview,
                "raw_sql": sql_text,
                "params": params,
                "insights": insights,
                "entities": plan.get("entities", {}),
                "chart": chart_spec,
                "handler_name": handler_name,
                "training_example_logged": should_learn,
            })

        except pyodbc.Error as err:
            err_text = str(err)
            LOGGER.warning(f"Custom SQL failed on attempt {attempt + 1}: {err_text}")
            if attempt < max_retries - 1:
                retry_reason = err_text
                continue

            missing_cols = extract_invalid_column_names(err_text)
            tables_in_sql = extract_table_tokens(sql_text)
            column_hints = summarize_table_columns(tables_in_sql, schema)

            detail_parts = [f"Database rejected the SQL: {err_text}"]
            if missing_cols:
                detail_parts.append(f"Columns not found: {', '.join(missing_cols)}.")
            if column_hints:
                detail_parts.append(f"Try columns from these tables instead: {column_hints}.")

            return _with_usage({"error": " ".join(detail_parts), "sql": sql_preview})
            
    return _with_usage({"error": "I was unable to generate a working query after multiple attempts."})
