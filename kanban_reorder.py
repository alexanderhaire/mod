"""
Kanban-Aware Integrated Reorder Engine

Implements the shop-floor rule:
    necessity -> location -> kanban

Open sales orders (SOP10100/SOP10200) are ranked by priority, exploded
through the BOM (BM010115) into raw-material demand, netted against on-hand
by location (MAIN first), netted against open POs (POP10110), and then a
kanban refill layer is added from ERP-derived monthly consumption rates
(IV30300 + WO010032/MOP10213).

This module is pure Python / pandas. Streamlit presentation lives in
pages/kanban_reorder.py.

Reuses:
    priority.calculate_priority_score          - ported from Production repo
    production_queries.fetch_open_orders_buckets
    inventory_queries.fetch_recursive_bom_for_item
    inventory_queries.fetch_on_hand_by_item
    inventory_queries.fetch_open_po_supply
    reorder_math.get_reorder_recommendations   - for vendor + lead time
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd
import pyodbc

from constants import PRIMARY_LOCATION, RAW_MATERIAL_CLASS_CODES
from priority import calculate_priority_score, urgency_label
from production_queries import fetch_open_orders_buckets
from inventory_queries import (
    fetch_recursive_bom_for_item,
    fetch_on_hand_by_item,
    fetch_open_po_supply,
    fetch_last_vendor_receipt_map,
    fetch_obsolete_item_set,
    fetch_dilution_proxy_map,
)
from synthetic_bom import reconstruct_synthetic_bom, MIN_CONFIDENT_MOS

# A raw-material leaf is "vendor-sourced" if it has a POP30310 receipt within
# this many days. Items outside this window are treated as manufactured-in-house
# and drilled via a synthetic BOM reconstructed from MOP1016 picklist history.
VENDOR_SOURCE_WINDOW_DAYS = 3 * 365

# Safety rail for the recursive drill into synthetic BOMs — a malformed MO
# history loop shouldn't take down the page.
MAX_DRILL_DEPTH = 5

LOGGER = logging.getLogger(__name__)

# Default mixer -> item prefix mapping, copied from
# c:/Users/alexh/Production/shared/constants.py:64-72.
# Hard-coded here so we don't depend on Production's SQLite config layer.
DEFAULT_MIXER_PREFIXES: dict[str, list[str]] = {
    "Phosphate": ["PHOS", "MAP", "MKP", "PHOSPHORIC", "GPCARBFULL"],
    "Glucohept": ["GLUCO", "CHELATE", "EDTA", "CHEL", "GOLD", "USC", "FLO", "SOARMIC"],
    "Magnum":    ["DYN", "MAGN", "BULK"],
    "Nitrate":   ["NO3", "NITRATE", "UREA", "NPK", "ZON"],
    "Mini tote": [],  # qty/format based, not prefix
    "Fertilizer": ["MAC"],
    "Liquid":    ["SOAR"],
}

# Fiscal year starts in October (inherited from the manual Kanban spreadsheet)
FISCAL_YEAR_START_MONTH = 10


# =============================================================================
# Helpers
# =============================================================================

def assign_mixer(item_number: str) -> str:
    """Return the mixer name for a finished-good item, or 'Unassigned'."""
    if not item_number:
        return "Unassigned"
    upper = item_number.upper()
    # Longer prefixes win - sort by length desc so 'PHOSPHORIC' beats 'PHOS'.
    best_match: tuple[str, int] = ("Unassigned", 0)
    for mixer, prefixes in DEFAULT_MIXER_PREFIXES.items():
        for pfx in prefixes:
            if upper.startswith(pfx) and len(pfx) > best_match[1]:
                best_match = (mixer, len(pfx))
    return best_match[0]


def _to_fiscal_month(calendar_month: int) -> int:
    """Map calendar month (1-12) to fiscal month (Oct=1, Nov=2, ..., Sep=12)."""
    return ((calendar_month - FISCAL_YEAR_START_MONTH) % 12) + 1


def _fiscal_month_label(calendar_month: int) -> str:
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return names[calendar_month - 1]


# =============================================================================
# 1. compute_kanban_rates
# =============================================================================

def compute_kanban_rates(
    cursor: pyodbc.Cursor,
    lookback_months: int = 24,
    location: str = PRIMARY_LOCATION,
) -> pd.DataFrame:
    """
    Derive ERP-sourced monthly consumption rates per item.

    Returns a DataFrame with one row per item_number:
        item_number, monthly_rate_avg, monthly_rate_p80,
        monthly_rate_last3, months_observed, monthly_history (list of dicts)

    The monthly_history column is a list of {calendar_month, fiscal_month,
    month_label, year, qty} records so the UI can render an Oct-Sep pivot
    matching the legacy Kanban spreadsheet.

    Uses IV30300 with TRXQTY < 0 (all outbound consumption), which captures
    both MO component issues AND sales. This is the total "drain" on
    inventory, which is what the kanban buffer protects against.
    """
    query = """
        SELECT
            ITEMNMBR,
            YEAR(DOCDATE)  AS Yr,
            MONTH(DOCDATE) AS Mo,
            SUM(ABS(TRXQTY)) AS Qty
        FROM IV30300
        WHERE TRXLOCTN = ?
          AND TRXQTY < 0
          AND DOCDATE >= DATEADD(month, ?, GETDATE())
        GROUP BY ITEMNMBR, YEAR(DOCDATE), MONTH(DOCDATE)
        ORDER BY ITEMNMBR, YEAR(DOCDATE), MONTH(DOCDATE)
    """
    try:
        cursor.execute(query, (location, -lookback_months))
        rows = cursor.fetchall()
    except pyodbc.Error as err:
        LOGGER.warning("compute_kanban_rates failed: %s", err)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    by_item: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        item = (r.ITEMNMBR or "").strip()
        if not item:
            continue
        by_item.setdefault(item, []).append({
            "year": int(r.Yr),
            "calendar_month": int(r.Mo),
            "fiscal_month": _to_fiscal_month(int(r.Mo)),
            "month_label": _fiscal_month_label(int(r.Mo)),
            "qty": float(r.Qty or 0),
        })

    out: list[dict[str, Any]] = []
    today = datetime.date.today()
    for item, history in by_item.items():
        qtys = [h["qty"] for h in history]
        if not qtys:
            continue
        qtys_sorted = sorted(qtys)
        p80_idx = max(0, int(len(qtys_sorted) * 0.80) - 1)
        monthly_rate_p80 = qtys_sorted[p80_idx] if qtys_sorted else 0.0

        # Last 3 months (by calendar) trailing
        three_ago = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
        two_ago = (three_ago - datetime.timedelta(days=1)).replace(day=1)
        start = (two_ago - datetime.timedelta(days=1)).replace(day=1)
        recent = [
            h["qty"] for h in history
            if datetime.date(h["year"], h["calendar_month"], 1) >= start
        ]
        monthly_rate_last3 = (sum(recent) / len(recent)) if recent else 0.0

        out.append({
            "item_number": item,
            "monthly_rate_avg": sum(qtys) / len(qtys),
            "monthly_rate_p80": monthly_rate_p80,
            "monthly_rate_last3": monthly_rate_last3,
            "months_observed": len(qtys),
            "monthly_history": history,
        })

    return pd.DataFrame(out)


# =============================================================================
# 2. fetch_open_demand_prioritized
# =============================================================================

def fetch_open_demand_prioritized(
    cursor: pyodbc.Cursor,
    today: datetime.date | None = None,
) -> pd.DataFrame:
    """
    Return open SOP demand with priority scoring applied.

    Output columns:
        item_number, item_description, total_qty, earliest_req_date,
        urgency_bucket, priority_score, urgency_label, driving_customer,
        driving_order
    """
    today = today or datetime.date.today()
    result = fetch_open_orders_buckets(cursor, today)
    buckets: dict = result.get("buckets", {})

    rows: list[dict[str, Any]] = []
    for bucket_name, entries in buckets.items():
        for e in entries:
            try:
                req_date = datetime.datetime.strptime(e["req_date"], "%Y-%m-%d").date()
            except (ValueError, KeyError):
                req_date = None
            score = calculate_priority_score(
                req_ship_date=req_date,
                today=today,
                total_qty=float(e.get("quantity") or 0),
            )
            rows.append({
                "order_number": e["order_number"],
                "customer": e["customer"],
                "req_date": req_date,
                "item_number": e["item_number"],
                "item_description": e["item_desc"],
                "quantity": float(e.get("quantity") or 0),
                "uofm": e.get("uofm", ""),
                "bucket": bucket_name,
                "priority_score": score,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "item_number", "item_description", "total_qty", "earliest_req_date",
            "urgency_bucket", "priority_score", "urgency_label",
            "driving_customer", "driving_order",
        ])

    df = pd.DataFrame(rows)

    # Aggregate per finished-good item
    agg = df.sort_values("priority_score", ascending=False).groupby("item_number").agg(
        item_description=("item_description", "first"),
        total_qty=("quantity", "sum"),
        earliest_req_date=("req_date", "min"),
        priority_score=("priority_score", "max"),
        urgency_bucket=("bucket", "first"),   # bucket of highest-priority line
        driving_customer=("customer", "first"),
        driving_order=("order_number", "first"),
    ).reset_index()

    agg["urgency_label"] = agg["priority_score"].apply(urgency_label)
    agg = agg.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return agg


# =============================================================================
# 3. explode_demand_to_raw_materials
# =============================================================================

def explode_demand_to_raw_materials(
    cursor: pyodbc.Cursor,
    demand_df: pd.DataFrame,
    progress_cb=None,
    today: datetime.date | None = None,
    obsolete_items: set[str] | None = None,
    last_receipt: dict[str, datetime.date | None] | None = None,
) -> dict[str, Any]:
    """
    Explode finished-good demand into raw-material requirements using the
    recursive BOM, with two post-leaf behaviors:

      * Items not purchased from a vendor in the past ``VENDOR_SOURCE_WINDOW_DAYS``
        are reclassified as "manufactured in-house" and drilled through a
        synthetic BOM reconstructed from the last 12 months of MOP1016 history.
      * Obsolete / discontinued items (see ``fetch_obsolete_item_set``) are
        dropped on the floor rather than propagated.

    Returns a dict so callers can surface diagnostics alongside the buy list:

        {
            "raw_needs": DataFrame,              # same shape as before
            "manufactured_report": DataFrame,    # items we synthesized
            "missing_bom_report": DataFrame,     # manufactured + no MO history
        }

    ``obsolete_items`` and ``last_receipt`` are accepted as pre-computed inputs
    so the orchestrator can share them with other stages and avoid re-querying.
    """
    empty_result = {
        "raw_needs": pd.DataFrame(),
        "manufactured_report": pd.DataFrame(),
        "missing_bom_report": pd.DataFrame(),
        "skipped_report": pd.DataFrame(),
    }
    if demand_df is None or demand_df.empty:
        return empty_result

    today = today or datetime.date.today()
    cutoff = today - datetime.timedelta(days=VENDOR_SOURCE_WINDOW_DAYS)

    if obsolete_items is None:
        obsolete_items = fetch_obsolete_item_set(cursor)
    if last_receipt is None:
        last_receipt = fetch_last_vendor_receipt_map(cursor)

    bom_cache: dict[str, list] = {}
    synth_cache: dict[str, list] = {}
    accumulator: dict[str, dict[str, Any]] = {}
    manufactured_seen: dict[str, dict[str, Any]] = {}
    missing_bom_seen: dict[str, dict[str, Any]] = {}
    skipped_seen: dict[str, dict[str, Any]] = {}

    def _classify(item: str) -> str:
        if item in obsolete_items:
            return "obsolete"
        rcpt = last_receipt.get(item)
        if rcpt is None or rcpt < cutoff:
            return "manufactured"
        return "vendor"

    def _accumulate(
        leaf: str,
        qty_required: float,
        driving_row: "pd.Series",
        driving_parent: str,
    ) -> None:
        existing = accumulator.get(leaf)
        priority = float(driving_row["priority_score"])
        if existing is None:
            accumulator[leaf] = {
                "item_number": leaf,
                "sop_derived_demand": qty_required,
                "max_priority_score": priority,
                "driving_parent": driving_parent,
                "driving_parent_desc": driving_row.get("item_description", ""),
                "driving_customer": driving_row.get("driving_customer", ""),
                "driving_bucket": driving_row.get("urgency_bucket", ""),
                "earliest_req_date": driving_row.get("earliest_req_date"),
            }
        else:
            existing["sop_derived_demand"] += qty_required
            if priority > existing["max_priority_score"]:
                existing["max_priority_score"] = priority
                existing["driving_parent"] = driving_parent
                existing["driving_parent_desc"] = driving_row.get("item_description", "")
                existing["driving_customer"] = driving_row.get("driving_customer", "")
                existing["driving_bucket"] = driving_row.get("urgency_bucket", "")
                existing["earliest_req_date"] = driving_row.get("earliest_req_date")

    def _drill(
        item: str,
        qty_required: float,
        driving_row: "pd.Series",
        driving_parent: str,
        depth: int,
    ) -> None:
        if depth > MAX_DRILL_DEPTH:
            LOGGER.warning("drill depth cap hit on %s (depth=%d)", item, depth)
            return

        cls = _classify(item)
        if cls == "obsolete":
            # Dropped: we never buy, make, or net against obsolete stock.
            return
        if cls == "vendor":
            _accumulate(item, qty_required, driving_row, driving_parent)
            return

        # Manufactured: synthesize a BOM from MO history and recurse.
        if item not in synth_cache:
            try:
                synth_cache[item] = reconstruct_synthetic_bom(cursor, item)
            except pyodbc.Error as err:
                LOGGER.warning("synthetic BOM failed for %s: %s", item, err)
                synth_cache[item] = []
                entry = missing_bom_seen.setdefault(item, {
                    "item_number": item,
                    "needed_qty": 0.0,
                    "driving_parent": driving_parent,
                    "driving_customer": driving_row.get("driving_customer", ""),
                    "reason": "synthetic_bom_error",
                    "error": str(err)[:200],
                })
                entry["needed_qty"] += qty_required
                return
        synth = synth_cache[item]

        if not synth:
            entry = missing_bom_seen.setdefault(item, {
                "item_number": item,
                "needed_qty": 0.0,
                "driving_parent": driving_parent,
                "driving_customer": driving_row.get("driving_customer", ""),
                "reason": "no_mo_history",
                "error": "",
            })
            entry["needed_qty"] += qty_required
            return

        # Record for the Manufactured Intermediates report.
        manu_entry = manufactured_seen.setdefault(item, {
            "item_number": item,
            "needed_qty": 0.0,
            "n_mos_observed": synth[0].n_mos_observed,
            "low_confidence": synth[0].n_mos_observed < MIN_CONFIDENT_MOS,
            "component_count": len(synth),
            "driving_parent": driving_parent,
            "driving_customer": driving_row.get("driving_customer", ""),
        })
        manu_entry["needed_qty"] += qty_required

        for sr in synth:
            comp = (sr.RawMaterial or "").strip()
            if not comp:
                continue
            comp_qty = float(sr.Design_Qty or 0) * qty_required
            if comp_qty <= 0:
                continue
            _drill(comp, comp_qty, driving_row, driving_parent, depth + 1)

    total_parents = len(demand_df)
    for i, (_, row) in enumerate(demand_df.iterrows()):
        parent = row["item_number"]
        qty = float(row["total_qty"] or 0)
        if qty <= 0:
            continue
        if parent in obsolete_items:
            # Obsolete FG on an open order — warn but don't drill.
            LOGGER.warning("obsolete item %s appears on an open order", parent)
            continue

        if progress_cb is not None:
            progress_cb(i + 1, total_parents, parent)

        try:
            if parent not in bom_cache:
                bom_rows, _ = fetch_recursive_bom_for_item(cursor, parent)
                bom_cache[parent] = bom_rows or []

            for b in bom_cache[parent]:
                raw_item = (b.RawMaterial or "").strip()
                if not raw_item:
                    continue
                design_qty = float(b.Design_Qty or 0)
                required = design_qty * qty
                if required <= 0:
                    continue
                _drill(raw_item, required, row, parent, depth=0)
        except pyodbc.Error as err:
            LOGGER.warning("BOM explosion failed for %s: %s", parent, err)
            skipped_seen[parent] = {
                "item_number": parent,
                "qty": qty,
                "reason": "query_timeout_or_error",
                "error": str(err)[:200],
                "driving_customer": row.get("driving_customer", ""),
            }
            continue

    raw_needs_df = pd.DataFrame(list(accumulator.values())) if accumulator else pd.DataFrame()
    manu_df = pd.DataFrame(list(manufactured_seen.values())) if manufactured_seen else pd.DataFrame()
    missing_df = pd.DataFrame(list(missing_bom_seen.values())) if missing_bom_seen else pd.DataFrame()
    skipped_df = pd.DataFrame(list(skipped_seen.values())) if skipped_seen else pd.DataFrame()

    return {
        "raw_needs": raw_needs_df,
        "manufactured_report": manu_df,
        "missing_bom_report": missing_df,
        "skipped_report": skipped_df,
    }


# =============================================================================
# 4. net_by_location
# =============================================================================

def _discover_locations(cursor: pyodbc.Cursor) -> list[str]:
    """Return distinct LOCNCODE values present in IV00102, MAIN first."""
    try:
        cursor.execute("SELECT DISTINCT RTRIM(LOCNCODE) AS LOC FROM IV00102")
        locs = [r.LOC for r in cursor.fetchall() if r.LOC]
    except pyodbc.Error as err:
        LOGGER.warning("_discover_locations failed: %s", err)
        return [PRIMARY_LOCATION]
    locs = sorted(set(locs))
    if PRIMARY_LOCATION in locs:
        locs.remove(PRIMARY_LOCATION)
    return [PRIMARY_LOCATION] + locs


def net_by_location(
    cursor: pyodbc.Cursor,
    needs_df: pd.DataFrame,
    location_priority: list[str] | None = None,
    dilution_proxies: dict[str, list[tuple[str, float]]] | None = None,
) -> pd.DataFrame:
    """
    Net gross requirements against on-hand by location (MAIN first) and open POs.

    Expects needs_df to have 'item_number' and 'gross_requirement' columns.
    Adds: on_hand_main, on_hand_other, on_order, proxy_credit, proxy_sources,
    net_need, spill_location.

    ``dilution_proxies`` maps a leaf item to a list of ``(parent_item, factor)``
    pairs where the parent is a dilution form (e.g. NO3FE → [(REC-NO3FE sibling)]).
    On-hand for the leaf is credited with ``parent_on_hand * factor``, which is
    the leaf-equivalent stock sitting under the parent code. This handles the
    case where vendors ship a concentrate (REC-NO3FE) that gets diluted on
    arrival into a different item code (NO3FE) but is still effectively the
    same inventory.
    """
    if needs_df is None or needs_df.empty:
        return needs_df

    if not location_priority:
        location_priority = _discover_locations(cursor)
    if PRIMARY_LOCATION not in location_priority:
        location_priority = [PRIMARY_LOCATION] + list(location_priority)

    items = needs_df["item_number"].tolist()

    # Also pull on-hand for every dilution parent we might need to credit.
    proxy_items: set[str] = set()
    if dilution_proxies:
        for leaf in items:
            for parent, _factor in dilution_proxies.get(leaf, []):
                proxy_items.add(parent)

    all_on_hand_items = list({*items, *proxy_items})

    on_hand_by_loc: dict[str, dict[str, float]] = {}
    for loc in location_priority:
        oh, _ = fetch_on_hand_by_item(cursor, all_on_hand_items, location=loc)
        on_hand_by_loc[loc] = {k: float(v) for k, v in oh.items()}

    on_order_map, _ = fetch_open_po_supply(cursor, items, location=PRIMARY_LOCATION)
    on_order_map = {k: float(v) for k, v in on_order_map.items()}

    enriched = needs_df.copy()
    on_hand_main = []
    on_hand_other = []
    on_order = []
    proxy_credit_col = []
    proxy_sources_col = []
    net_need = []
    spill_location: list[str] = []

    for _, row in enriched.iterrows():
        item = row["item_number"]
        gross = float(row.get("gross_requirement") or 0)

        main_qty = on_hand_by_loc.get(PRIMARY_LOCATION, {}).get(item, 0.0)
        other_qty = 0.0
        spill = ""
        for loc in location_priority:
            if loc == PRIMARY_LOCATION:
                continue
            q = on_hand_by_loc.get(loc, {}).get(item, 0.0)
            if q > 0:
                other_qty += q
                if not spill:
                    spill = loc

        po_qty = on_order_map.get(item, 0.0)

        # Dilution-proxy credit: sum leaf-equivalent stock sitting under any
        # parent item that is effectively the same material. Credit across
        # ALL locations, since the dilution process happens at MAIN anyway.
        proxy_credit = 0.0
        proxy_sources: list[str] = []
        if dilution_proxies:
            for parent, factor in dilution_proxies.get(item, []):
                parent_total = 0.0
                for loc in location_priority:
                    parent_total += on_hand_by_loc.get(loc, {}).get(parent, 0.0)
                if parent_total > 0:
                    proxy_credit += parent_total * factor
                    proxy_sources.append(f"{parent}×{factor:.3f}")

        # Pull MAIN first, then spill locations, then proxy credit, then POs.
        remaining = gross - main_qty
        if remaining > 0:
            remaining -= other_qty
        else:
            other_qty = 0.0
        if remaining > 0:
            remaining -= proxy_credit
        else:
            proxy_credit = 0.0
        if remaining > 0:
            remaining -= po_qty
        need = max(0.0, remaining)

        on_hand_main.append(main_qty)
        on_hand_other.append(other_qty)
        on_order.append(po_qty)
        proxy_credit_col.append(proxy_credit)
        proxy_sources_col.append(", ".join(proxy_sources))
        net_need.append(need)
        spill_location.append(spill)

    enriched["on_hand_main"] = on_hand_main
    enriched["on_hand_other"] = on_hand_other
    enriched["on_order"] = on_order
    enriched["proxy_credit"] = proxy_credit_col
    enriched["proxy_sources"] = proxy_sources_col
    enriched["net_need"] = net_need
    enriched["spill_location"] = spill_location
    return enriched


# =============================================================================
# 5. build_integrated_reorder_list (orchestrator)
# =============================================================================

def build_integrated_reorder_list(
    cursor: pyodbc.Cursor,
    today: datetime.date | None = None,
    lookback_months: int = 24,
    include_future_demand: bool = False,
    location_priority: list[str] | None = None,
    stage_cb=None,
    bom_progress_cb=None,
) -> dict[str, Any]:
    """
    Top-level orchestrator. Returns:
        {
            "raw_materials": pd.DataFrame,  # buy list
            "finished_goods": pd.DataFrame, # production queue
            "kanban_rates": pd.DataFrame,   # monthly pivot source
            "as_of": date,
        }

    Optional callbacks for progress reporting:
        stage_cb(stage_index, total_stages, label) - called at the start of each stage
        bom_progress_cb(current, total, item_name) - called inside the BOM explosion loop
    """
    today = today or datetime.date.today()
    TOTAL_STAGES = 6

    def _stage(i: int, label: str):
        if stage_cb is not None:
            stage_cb(i, TOTAL_STAGES, label)

    # Shared lookups used by multiple stages — pulled once so the filter,
    # explode, and netting stages all see a consistent view of the data.
    obsolete_items = fetch_obsolete_item_set(cursor)
    last_receipt = fetch_last_vendor_receipt_map(cursor)
    dilution_proxies = fetch_dilution_proxy_map(cursor)

    # 1. Kanban rates from ERP
    _stage(1, "Computing kanban rates from 24 months of IV30300...")
    rates_df = compute_kanban_rates(cursor, lookback_months=lookback_months)

    # 2. Open demand, prioritized. Drop obsolete FGs before they fan out
    # through the BOM explosion — they should never appear on the buy list,
    # the production queue, or the diagnostics tabs.
    _stage(2, "Fetching and prioritizing open sales orders...")
    demand_df = fetch_open_demand_prioritized(cursor, today)
    if not include_future_demand and not demand_df.empty:
        demand_df = demand_df[demand_df["urgency_bucket"] != "future"].copy()
    if not demand_df.empty and obsolete_items:
        before = len(demand_df)
        demand_df = demand_df[~demand_df["item_number"].isin(obsolete_items)].copy()
        dropped = before - len(demand_df)
        if dropped:
            LOGGER.info("dropped %d obsolete finished goods from demand", dropped)

    # 3. Explode to raw materials (leaf-only BOM walk + manufactured-item
    # re-drill via synthetic BOMs). Returns a dict with the buy-list dataframe
    # plus two diagnostic reports.
    _stage(3, f"Exploding BOM for {len(demand_df)} finished goods...")
    explode_result = explode_demand_to_raw_materials(
        cursor,
        demand_df,
        progress_cb=bom_progress_cb,
        today=today,
        obsolete_items=obsolete_items,
        last_receipt=last_receipt,
    )
    raw_needs = explode_result["raw_needs"]
    manufactured_report = explode_result["manufactured_report"]
    missing_bom_report = explode_result["missing_bom_report"]
    skipped_report = explode_result.get("skipped_report", pd.DataFrame())

    # 4. Merge kanban refill layer onto raw needs (and vice-versa: kanban-only items)
    rates_min = rates_df[["item_number", "monthly_rate_p80", "monthly_rate_last3",
                          "monthly_rate_avg"]] if not rates_df.empty else pd.DataFrame(
        columns=["item_number", "monthly_rate_p80", "monthly_rate_last3", "monthly_rate_avg"]
    )

    if raw_needs.empty:
        raw_needs = pd.DataFrame(columns=[
            "item_number", "sop_derived_demand", "max_priority_score",
            "driving_parent", "driving_parent_desc", "driving_customer",
            "driving_bucket", "earliest_req_date",
        ])

    merged = raw_needs.merge(rates_min, on="item_number", how="outer")
    merged["sop_derived_demand"] = merged["sop_derived_demand"].fillna(0.0)
    merged["monthly_rate_p80"] = merged["monthly_rate_p80"].fillna(0.0)
    merged["monthly_rate_last3"] = merged["monthly_rate_last3"].fillna(0.0)
    merged["monthly_rate_avg"] = merged["monthly_rate_avg"].fillna(0.0)
    merged["max_priority_score"] = merged["max_priority_score"].fillna(0.0)
    merged["driving_parent"] = merged["driving_parent"].fillna("")
    merged["driving_parent_desc"] = merged["driving_parent_desc"].fillna("")
    merged["driving_customer"] = merged["driving_customer"].fillna("")
    merged["driving_bucket"] = merged["driving_bucket"].fillna("kanban_only")

    # 5. Filter to raw materials only (skip finished goods that happen to appear
    # in BOM self-loops or kanban rates). Use class code via an IV00101 lookup.
    _stage(4, "Filtering to raw materials and merging kanban rates...")
    merged = _filter_raw_materials(cursor, merged)

    if merged.empty:
        raw_out = merged
    else:
        # 6. Kanban refill qty layer: one month of rate (conservative default).
        # The UI can multiply by lead_time/30 if desired.
        merged["kanban_refill_qty"] = merged["monthly_rate_p80"]
        merged["gross_requirement"] = merged["sop_derived_demand"] + merged["kanban_refill_qty"]

        # 7. Net by location + on-order, with dilution-proxy on-hand credit
        # so e.g. NO3FE stock counts toward REC-NO3FE requirements.
        _stage(5, "Netting against MAIN, spill locations, and open POs...")
        merged = net_by_location(
            cursor,
            merged,
            location_priority=location_priority,
            dilution_proxies=dilution_proxies,
        )

        # 8. Urgency label (kanban-only items get a LOW label)
        merged["urgency_label"] = merged["max_priority_score"].apply(
            lambda s: urgency_label(float(s)) if s and s > 0 else "KANBAN"
        )

        # 9. Only keep items that actually need buying
        merged = merged[merged["net_need"] > 0].copy()
        merged = merged.sort_values(
            ["max_priority_score", "net_need"], ascending=[False, False]
        ).reset_index(drop=True)
        raw_out = merged

    # 10. Finished goods production queue (SOP demand with mixer assignment
    # and kanban top-off suggestion)
    _stage(6, "Building finished-goods production queue...")
    if demand_df.empty:
        fg_out = demand_df
    else:
        fg = demand_df.copy()
        fg["mixer"] = fg["item_number"].apply(assign_mixer)
        # Kanban top-off for each finished good: suggest buffer refill on top
        # of the order qty so a mixer run covers both the sale and the shelf.
        fg = fg.merge(
            rates_df[["item_number", "monthly_rate_p80"]] if not rates_df.empty
            else pd.DataFrame(columns=["item_number", "monthly_rate_p80"]),
            on="item_number", how="left",
        )
        fg["monthly_rate_p80"] = fg["monthly_rate_p80"].fillna(0.0)
        fg["kanban_extra_qty"] = (fg["monthly_rate_p80"] - fg["total_qty"]).clip(lower=0.0)
        fg["suggested_batch_qty"] = fg["total_qty"] + fg["kanban_extra_qty"]
        fg_out = fg

    return {
        "raw_materials": raw_out,
        "finished_goods": fg_out,
        "kanban_rates": rates_df,
        "manufactured_report": manufactured_report,
        "missing_bom_report": missing_bom_report,
        "skipped_report": skipped_report,
        "as_of": today,
    }


def _filter_raw_materials(cursor: pyodbc.Cursor, df: pd.DataFrame) -> pd.DataFrame:
    """Keep only items whose ITMCLSCD is in RAW_MATERIAL_CLASS_CODES."""
    if df.empty:
        return df
    items = df["item_number"].dropna().unique().tolist()
    if not items:
        return df
    placeholders = ", ".join("?" for _ in items)
    class_placeholders = ", ".join("?" for _ in RAW_MATERIAL_CLASS_CODES)
    query = f"""
        SELECT ITEMNMBR, ITEMDESC, ITMCLSCD
        FROM IV00101
        WHERE ITEMNMBR IN ({placeholders})
          AND ITMCLSCD IN ({class_placeholders})
    """
    try:
        cursor.execute(query, (*items, *RAW_MATERIAL_CLASS_CODES))
        rows = cursor.fetchall()
    except pyodbc.Error as err:
        LOGGER.warning("_filter_raw_materials failed: %s", err)
        return df

    keep = {(r.ITEMNMBR or "").strip(): {
        "description": (r.ITEMDESC or "").strip(),
        "class": (r.ITMCLSCD or "").strip(),
    } for r in rows}

    df = df[df["item_number"].isin(keep.keys())].copy()
    df["item_description"] = df["item_number"].map(lambda i: keep[i]["description"])
    df["item_class"] = df["item_number"].map(lambda i: keep[i]["class"])
    return df
