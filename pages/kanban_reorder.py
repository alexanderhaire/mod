"""
Kanban Reorder Page

Streamlit page that renders the integrated necessity -> location -> kanban
reorder list built by kanban_reorder.py.
"""

import datetime
import logging

import pandas as pd
import streamlit as st

from db_pool import get_connection as get_pooled_connection
from kanban_reorder import build_integrated_reorder_list

LOGGER = logging.getLogger(__name__)

_FISCAL_MONTH_ORDER = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
                       "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def render_kanban_reorder():
    st.header("🔁 Kanban Reorder — Integrated Buy + Production List")
    st.caption(
        "Necessity → Location → Kanban. "
        "Open sales orders drive raw-material needs, netted by location, "
        "then topped off with ERP-derived kanban buffer rates."
    )

    with st.sidebar:
        st.subheader("⚙️ Kanban Settings")
        lookback_months = st.selectbox(
            "Kanban Lookback (months)",
            options=[12, 24, 36],
            index=1,
            help="History window for deriving monthly consumption rates from IV30300",
        )
        include_future = st.checkbox(
            "Include 'future' SOP demand",
            value=False,
            help="By default only past due / today / tomorrow orders drive raw-material demand",
        )

    # Progress bar + status line. The orchestrator has 6 stages; inside stage 3
    # (BOM explosion) we show sub-progress per finished good, which is the slow
    # step because it runs one recursive CTE per unique parent item.
    progress_bar = st.progress(0.0, text="Starting...")
    status_line = st.empty()

    # Use mutable state so the nested closures can update a "current stage"
    # index and blend stage progress with sub-progress from the BOM loop.
    state = {"stage": 0, "stage_total": 6, "stage_label": ""}

    def _blend(sub_frac: float = 0.0) -> float:
        # Each stage owns 1/stage_total of the bar. Sub-fraction fills within
        # the current stage's slice.
        stage_slice = 1.0 / state["stage_total"]
        base = max(0, state["stage"] - 1) * stage_slice
        return min(1.0, base + stage_slice * max(0.0, min(1.0, sub_frac)))

    def stage_cb(stage_idx: int, total: int, label: str):
        state["stage"] = stage_idx
        state["stage_total"] = total
        state["stage_label"] = label
        progress_bar.progress(_blend(0.0), text=f"Step {stage_idx}/{total}: {label}")
        status_line.caption(label)

    def bom_progress_cb(current: int, total: int, item_name: str):
        if total <= 0:
            return
        frac = current / total
        progress_bar.progress(
            _blend(frac),
            text=f"Step {state['stage']}/{state['stage_total']}: exploding BOM {current}/{total} ({item_name})",
        )

    try:
        with get_pooled_connection() as conn:
            conn.timeout = 60  # pyodbc query timeout — aborts runaway recursive CTEs
            cur = conn.cursor()
            result = build_integrated_reorder_list(
                cur,
                today=datetime.date.today(),
                lookback_months=lookback_months,
                include_future_demand=include_future,
                stage_cb=stage_cb,
                bom_progress_cb=bom_progress_cb,
            )
    except Exception as e:
        progress_bar.empty()
        status_line.empty()
        st.error(f"Error building integrated list: {e}")
        LOGGER.exception("build_integrated_reorder_list failed")
        return

    progress_bar.progress(1.0, text="Done ✅")
    status_line.empty()
    progress_bar.empty()

    raw_df: pd.DataFrame = result.get("raw_materials", pd.DataFrame())
    fg_df: pd.DataFrame = result.get("finished_goods", pd.DataFrame())
    rates_df: pd.DataFrame = result.get("kanban_rates", pd.DataFrame())
    manufactured_df: pd.DataFrame = result.get("manufactured_report", pd.DataFrame())
    missing_bom_df: pd.DataFrame = result.get("missing_bom_report", pd.DataFrame())
    skipped_df: pd.DataFrame = result.get("skipped_report", pd.DataFrame())
    as_of = result.get("as_of", datetime.date.today())

    if not skipped_df.empty:
        st.warning(
            f"⚠️ {len(skipped_df)} finished good(s) skipped due to slow/failed BOM queries. "
            "Buy list below is incomplete for those items — see the 'BOM Missing' tab."
        )

    _render_header_metrics(raw_df, fg_df, as_of)

    st.divider()

    tab_raw, tab_fg, tab_rates, tab_top, tab_manu, tab_missing = st.tabs([
        "🧪 Raw Materials (Buy List)",
        "🏭 Finished Goods (Production Queue)",
        "📅 Kanban Rates (Oct–Sep)",
        "🏆 Top Sellers (12-mo)",
        "🛠 Manufactured Intermediates",
        "⚠️ BOM Missing",
    ])

    with tab_raw:
        _render_raw_materials(raw_df)
    with tab_fg:
        _render_finished_goods(fg_df)
    with tab_rates:
        _render_kanban_rates(rates_df)
    with tab_top:
        _render_top_sellers()
    with tab_manu:
        _render_manufactured_intermediates(manufactured_df)
    with tab_missing:
        _render_missing_bom(missing_bom_df, skipped_df)


# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------

def _render_header_metrics(raw_df: pd.DataFrame, fg_df: pd.DataFrame, as_of: datetime.date):
    col1, col2, col3, col4 = st.columns(4)

    past_due = 0
    if not fg_df.empty and "urgency_bucket" in fg_df.columns:
        past_due = int((fg_df["urgency_bucket"] == "past_due").sum())

    raw_short = 0 if raw_df.empty else int(len(raw_df))
    fg_to_make = 0 if fg_df.empty else int(len(fg_df))

    with col1:
        st.metric("🚨 Past-Due Orders", past_due, help="Open SOP lines where REQSHIPDATE < today")
    with col2:
        st.metric("🧪 Raw Materials Short", raw_short, help="Raw materials where net_need > 0 after MAIN + other locations + open POs")
    with col3:
        st.metric("🏭 Finished Goods to Produce", fg_to_make, help="Finished goods with open demand that should be scheduled")
    with col4:
        st.metric("📅 As Of", as_of.strftime("%Y-%m-%d"))


# ---------------------------------------------------------------------------
# Raw Materials tab
# ---------------------------------------------------------------------------

def _render_raw_materials(df: pd.DataFrame):
    if df.empty:
        st.info("No raw materials need ordering.")
        return

    display_cols = [
        "item_number", "item_description", "urgency_label",
        "sop_derived_demand", "kanban_refill_qty", "gross_requirement",
        "on_hand_main", "on_hand_other", "proxy_credit", "proxy_sources",
        "on_order", "net_need",
        "driving_parent", "driving_customer", "driving_bucket",
        "earliest_req_date", "spill_location", "max_priority_score",
    ]

    def _format_df(sub: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in display_cols if c in sub.columns]
        out = sub[present].copy()
        for c in ["sop_derived_demand", "kanban_refill_qty", "gross_requirement",
                   "on_hand_main", "on_hand_other", "proxy_credit", "on_order",
                   "net_need", "max_priority_score"]:
            if c in out.columns:
                out[c] = out[c].astype(float).round(2)
        return out

    def _row_color(row):
        bucket = row.get("driving_bucket", "")
        if bucket == "past_due":
            return ["background-color: #ffcccc"] * len(row)
        if bucket == "due_today":
            return ["background-color: #ffe0b3"] * len(row)
        if bucket == "due_tomorrow":
            return ["background-color: #fff4b3"] * len(row)
        if bucket == "kanban_only":
            return ["background-color: #e0ebff"] * len(row)
        return [""] * len(row)

    def _show(sub: pd.DataFrame):
        display_df = _format_df(sub)
        try:
            styled = display_df.style.apply(_row_color, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Split into SOP-driven (urgent) vs kanban-only (buffer restock)
    has_bucket = "driving_bucket" in df.columns
    if has_bucket:
        sop_mask = df["driving_bucket"] != "kanban_only"
        sop_df = df[sop_mask]
        kanban_df = df[~sop_mask]
    else:
        sop_df = df
        kanban_df = df.iloc[0:0]

    # --- Section 1: Order-driven ---
    st.subheader(f"Order NOW — open-order demand ({len(sop_df)} items)")
    st.caption(
        "Raw materials required by open sales orders (past-due, due today, due tomorrow). "
        "These drive today's production schedule."
    )
    if sop_df.empty:
        st.success("All open-order raw materials are covered by on-hand + on-order supply.")
    else:
        _show(sop_df)

    # --- Section 2: Kanban buffer ---
    with st.expander(f"Restock — below kanban buffer ({len(kanban_df)} items)", expanded=False):
        st.caption(
            "Raw materials with no open-order demand but on-hand is below one month's "
            "P80 consumption rate. These are buffer maintenance, not urgent."
        )
        if kanban_df.empty:
            st.info("All kanban buffers are healthy.")
        else:
            _show(kanban_df)

    _render_csv_download(_format_df(df), "kanban_raw_materials.csv")


# ---------------------------------------------------------------------------
# Finished Goods tab
# ---------------------------------------------------------------------------

def _render_finished_goods(df: pd.DataFrame):
    if df.empty:
        st.info("No open sales-order demand for finished goods.")
        return

    st.subheader("Finished Goods Production Queue")
    st.caption("Each row is a finished good that has open SOP demand. `suggested_batch_qty` = order qty + kanban top-off.")

    display_cols = [
        "item_number", "item_description", "mixer", "urgency_label",
        "total_qty", "kanban_extra_qty", "suggested_batch_qty",
        "earliest_req_date", "driving_customer", "driving_order",
        "urgency_bucket", "priority_score",
    ]
    present = [c for c in display_cols if c in df.columns]
    display_df = df[present].copy()
    for c in ["total_qty", "kanban_extra_qty", "suggested_batch_qty", "priority_score"]:
        if c in display_df.columns:
            display_df[c] = display_df[c].astype(float).round(2)

    # Filter by mixer
    if "mixer" in display_df.columns:
        mixers = ["(all)"] + sorted(display_df["mixer"].unique().tolist())
        chosen = st.selectbox("Filter by mixer", mixers, index=0)
        if chosen != "(all)":
            display_df = display_df[display_df["mixer"] == chosen]

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    _render_csv_download(display_df, "kanban_finished_goods.csv")


# ---------------------------------------------------------------------------
# Kanban Rates tab
# ---------------------------------------------------------------------------

def _render_kanban_rates(df: pd.DataFrame):
    if df.empty:
        st.info("No kanban rate data available (no IV30300 consumption in window).")
        return

    st.subheader("Kanban Monthly Rates (Fiscal Year: Oct → Sep)")
    st.caption(
        "ERP-derived monthly consumption per item. "
        "`p80` is the 80th-percentile month (conservative buffer size)."
    )

    summary_cols = ["item_number", "months_observed",
                    "monthly_rate_avg", "monthly_rate_p80", "monthly_rate_last3"]
    summary = df[summary_cols].copy()
    for c in ["monthly_rate_avg", "monthly_rate_p80", "monthly_rate_last3"]:
        summary[c] = summary[c].astype(float).round(2)
    summary = summary.sort_values("monthly_rate_p80", ascending=False)

    st.markdown("**Summary**")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("**Oct→Sep Pivot (latest observed month per fiscal slot)**")
    pivot = _build_fiscal_pivot(df)
    if pivot is None or pivot.empty:
        st.info("Not enough history to build a full fiscal pivot.")
    else:
        st.dataframe(pivot, use_container_width=True)

    _render_csv_download(summary, "kanban_rates_summary.csv")


def _build_fiscal_pivot(df: pd.DataFrame) -> pd.DataFrame | None:
    """Build a pivot of item_number x Oct..Sep using the most recent qty per slot."""
    if df.empty or "monthly_history" not in df.columns:
        return None

    rows: list[dict] = []
    for _, r in df.iterrows():
        item = r["item_number"]
        hist = r.get("monthly_history") or []
        # For each calendar month, keep the most recent year's value
        latest: dict[str, tuple[int, float]] = {}
        for h in hist:
            label = h.get("month_label")
            year = h.get("year", 0)
            qty = h.get("qty", 0.0)
            if label not in latest or year > latest[label][0]:
                latest[label] = (year, qty)
        row_out = {"item_number": item}
        for m in _FISCAL_MONTH_ORDER:
            row_out[m] = round(latest[m][1], 2) if m in latest else 0.0
        rows.append(row_out)

    if not rows:
        return None
    pivot = pd.DataFrame(rows)
    pivot = pivot.set_index("item_number")
    pivot = pivot[_FISCAL_MONTH_ORDER]
    pivot = pivot.sort_values(_FISCAL_MONTH_ORDER, ascending=False)
    return pivot.reset_index()


# ---------------------------------------------------------------------------
# Top Sellers tab
# ---------------------------------------------------------------------------

def _render_top_sellers():
    st.subheader("Top Sellers — 12-Month Rolling")
    st.caption("From SOP30300/SOP30200 invoice history. Mirrors Sheet2 of the legacy Kanban spreadsheet.")

    query = """
        SELECT TOP 100
            d.ITEMNMBR,
            MAX(d.ITEMDESC) AS ITEMDESC,
            SUM(d.QTYFULFI) AS TotalQty,
            SUM(d.XTNDPRCE) AS TotalSales
        FROM SOP30300 d
        JOIN SOP30200 h ON d.SOPNUMBE = h.SOPNUMBE AND d.SOPTYPE = h.SOPTYPE
        WHERE h.SOPTYPE = 3
          AND h.DOCDATE >= DATEADD(year, -1, GETDATE())
          AND d.ITEMNMBR NOT LIKE 'FREIGHT%'
        GROUP BY d.ITEMNMBR
        ORDER BY TotalSales DESC
    """
    try:
        with get_pooled_connection() as conn:
            df = pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Top sellers query failed: {e}")
        return

    if df.empty:
        st.info("No invoice history in the last 12 months.")
        return

    df["TotalQty"] = df["TotalQty"].astype(float).round(2)
    df["TotalSales"] = df["TotalSales"].astype(float).round(2)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Manufactured Intermediates tab
# ---------------------------------------------------------------------------

def _render_manufactured_intermediates(df: pd.DataFrame):
    st.subheader("Manufactured Intermediates")
    st.caption(
        "Items classified as **made in-house** (no vendor purchase in the last 3 years). "
        "The buy list shows the raw materials that go INTO these items, not the items themselves. "
        "The recipe is reverse-engineered from the last 12 months of MO picklist history (MOP1016)."
    )
    if df.empty:
        st.info("No manufactured intermediates surfaced in this run.")
        return

    display_cols = [
        "item_number", "needed_qty", "n_mos_observed", "low_confidence",
        "component_count", "driving_parent", "driving_customer",
    ]
    present = [c for c in display_cols if c in df.columns]
    out = df[present].copy()
    if "needed_qty" in out.columns:
        out["needed_qty"] = out["needed_qty"].astype(float).round(2)
    out = out.sort_values("needed_qty", ascending=False) if "needed_qty" in out.columns else out
    st.dataframe(out, use_container_width=True, hide_index=True)
    _render_csv_download(out, "kanban_manufactured_intermediates.csv")


# ---------------------------------------------------------------------------
# BOM Missing tab
# ---------------------------------------------------------------------------

def _render_missing_bom(df: pd.DataFrame, skipped_df: pd.DataFrame | None = None):
    st.subheader("Manufactured — BOM Missing")
    st.caption(
        "Items that are *not* vendor-sourced AND have no MO picklist history to synthesize a BOM from. "
        "These items silently distort the buy list if ignored — investigate each one."
    )
    if df.empty:
        st.success("No items in this state. 🎉")
    else:
        display_cols = ["item_number", "needed_qty", "reason", "error",
                        "driving_parent", "driving_customer"]
        present = [c for c in display_cols if c in df.columns]
        out = df[present].copy()
        if "needed_qty" in out.columns:
            out["needed_qty"] = out["needed_qty"].astype(float).round(2)
            out = out.sort_values("needed_qty", ascending=False)
        st.dataframe(out, use_container_width=True, hide_index=True)
        _render_csv_download(out, "kanban_bom_missing.csv")

    if skipped_df is not None and not skipped_df.empty:
        st.divider()
        st.subheader("⏱ Skipped — Slow or Failed BOM Query")
        st.caption(
            "Finished goods whose BOM explosion timed out or errored. Their raw-material "
            "demand is NOT reflected in the buy list. Investigate GP side (slow recursive "
            "CTE, missing MOP1016 index, or a BOM cycle)."
        )
        skipped_cols = ["item_number", "qty", "reason", "error", "driving_customer"]
        present = [c for c in skipped_cols if c in skipped_df.columns]
        out = skipped_df[present].copy()
        if "qty" in out.columns:
            out["qty"] = out["qty"].astype(float).round(2)
        st.dataframe(out, use_container_width=True, hide_index=True)
        _render_csv_download(out, "kanban_bom_skipped.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_csv_download(df: pd.DataFrame, filename: str):
    if df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


if __name__ == "__main__":
    render_kanban_reorder()
