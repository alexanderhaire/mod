
import datetime
import re
from decimal import Decimal

import pyodbc

from constants import CUSTOM_SQL_MAX_ROWS, LOGGER, RAW_MATERIAL_CATEGORIES, RAW_MATERIAL_PREFIXES
from sql_utils import format_sql_preview


def handle_vendor_scorecard(cursor: pyodbc.Cursor, prompt: str, today: datetime.date) -> dict | None:
    """
    Generate a vendor scorecard showing spend, reliability, and variance.
    Handles prompts like "Show me a scorecard for Vendor X" or "Analyze our top vendors".
    """
    lower = prompt.lower()
    scorecard_tokens = ("scorecard", "vendor performance", "supplier performance", "analyze vendor", "vendor evaluation")
    if not any(tok in lower for tok in scorecard_tokens):
        return None

    # Try to extract a specific vendor name or ID
    # This is a basic extraction; for robust usage, we might need a dedicated entity extractor
    # or rely on the user providing a clear ID.
    vendor_pattern = r"vendor\s+([a-zA-Z0-9\s\-\.]+)"
    match = re.search(vendor_pattern, lower)
    vendor_filter = ""
    vendor_param = []
    
    # Check if asking for "top" or "all" instead of a specific vendor
    is_top_request = "top" in lower or "rank" in lower or "best" in lower or "worst" in lower
    
    specific_vendor_name = None
    if match and not is_top_request:
        raw_vendor = match.group(1).strip()
        # Clean up some common trailing words if the regex grabbed too much
        blacklist = ("scorecard", "performance", "analysis", "report")
        filet_vendor = raw_vendor
        for word in blacklist:
            filet_vendor = filet_vendor.replace(word, "").strip()
            
        if filet_vendor:
            # We'll use a LIKE search for flexibility
            vendor_filter = "AND (pm.VENDORID LIKE ? OR pm.VENDNAME LIKE ?)"
            search_term = f"%{filet_vendor}%"
            vendor_param = [search_term, search_term]
            specific_vendor_name = filet_vendor

    # Time window: default to last 12 months
    start_date = today - datetime.timedelta(days=365)
    
    # Query: Join Vendor Master (PM00200) with Purchase Order History (POP30100/POP30110)
    # Filter: Restrict to raw materials using prefixes/categories
    rm_category_list = "', '".join(RAW_MATERIAL_CATEGORIES)
    rm_prefix_conditions = " OR ".join([f"i.ITEMNMBR LIKE '{p}%'" for p in RAW_MATERIAL_PREFIXES])
    
    query = f"""
        WITH VendorStats AS (
            SELECT 
                h.VENDORID,
                COUNT(DISTINCT h.PONUMBER) as POCount,
                SUM(l.EXTDCOST) as TotalSpend,
                SUM(CASE WHEN h.RECEIPTDATE <= l.PRMSHPDTE THEN 1.0 ELSE 0.0 END) as OnTimeLines,
                COUNT(l.PONUMBER) as TotalLines
            FROM POP30100 h
            JOIN POP30110 l ON h.PONUMBER = l.PONUMBER
            LEFT JOIN IV00101 i ON l.ITEMNMBR = i.ITEMNMBR
            WHERE h.RECEIPTDATE >= ?
            AND (
                i.ITMCLSCD IN ('{rm_category_list}') 
                OR {rm_prefix_conditions}
            )
            GROUP BY h.VENDORID
        )
        SELECT TOP {CUSTOM_SQL_MAX_ROWS}
            pm.VENDORID,
            pm.VENDNAME,
            ISNULL(vs.POCount, 0) as POCount,
            ISNULL(vs.TotalSpend, 0) as TotalSpend,
            CASE 
                WHEN ISNULL(vs.TotalLines, 0) > 0 
                THEN (vs.OnTimeLines / vs.TotalLines) * 100 
                ELSE NULL 
            END as OnTimePct
        FROM PM00200 pm
        LEFT JOIN VendorStats vs ON pm.VENDORID = vs.VENDORID
        WHERE (vs.TotalSpend > 0 OR vs.POCount > 0)
        {vendor_filter}
        ORDER BY vs.TotalSpend DESC
    """
    
    params = [start_date] + vendor_param
    sql_preview = format_sql_preview(query, params)
    
    try:
        cursor.execute(query, params)
        fetched = cursor.fetchmany(CUSTOM_SQL_MAX_ROWS + 1)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, r)) for r in fetched[:CUSTOM_SQL_MAX_ROWS]]
        truncated = len(fetched) > CUSTOM_SQL_MAX_ROWS
    except pyodbc.Error as err:
        return {"error": f"Failed to generate vendor scorecard: {err}"}

    if not rows:
        msg = f"No vendor activity found in the last year."
        if specific_vendor_name:
            msg += f" (Searched for '{specific_vendor_name}')"
        return {"insights": {"summary": msg}, "sql": sql_preview}

    # Generate insights summary
    if specific_vendor_name and len(rows) == 1:
        row = rows[0]
        on_time = f"{row['OnTimePct']:.1f}%" if row['OnTimePct'] is not None else "N/A"
        summary = (
            f"**Vendor Scorecard for {row['VENDNAME']} ({row['VENDORID']})**\n\n"
            f"- **Total Spend (Last 12mo):** ${row['TotalSpend']:,.2f}\n"
            f"- **PO Count:** {row['POCount']}\n"
            f"- **On-Time Delivery:** {on_time} (based on receipt date vs promised date)"
        )
    else:
        summary = (
            f"**Top Vendor Scorecard (Last 12 Months)**\n"
            f"Found {len(rows)} vendors active in the last year. \n"
            f"Ranked by total spend."
        )

    insights = {
        "summary": summary, 
        "row_count": len(rows),
        "truncated": truncated
    }
    
    entities = {
        "intent": "vendor_scorecard",
        "vendor": specific_vendor_name if specific_vendor_name else "all"
    }

    return {"data": rows, "insights": insights, "sql": sql_preview, "entities": entities}
