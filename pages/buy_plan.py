"""
Buy Plan (Validated)

A focused, decision-oriented raw-material buy list built on the corrected demand
chain (open sales orders -> BOM -> raw materials, netted by location + open POs).

Quantities use the fixed BOM explosion (active recipe only; raw materials treated
as hard leaves) -- see inventory_queries.fetch_recursive_bom_for_item and
docs/superpowers/audits/2026-06-05-demand-chain-trust-report.md. Before that fix
this list over-stated raw demand 11x-23x by summing every archived batch recipe.

This page also QUARANTINES the two known phantom-buy classes the audit found --
REC- receiving codes (stock sits under the base code) and items with no vendor
purchase history (made in-house, not orderable) -- into a separate review section
so the main buy list stays trustworthy. It surfaces demand that did not explode
(orders with no BOM) instead of letting it vanish silently.
"""

import datetime
import logging

import pandas as pd
import streamlit as st

from db_pool import get_connection as get_pooled_connection
from kanban_reorder import build_integrated_reorder_list

LOGGER = logging.getLogger(__name__)

# Open-order urgency buckets, most-urgent first. Anything in these is demand we
# are committed to within the next few days (well inside a 17-day order horizon).
_ORDER_DRIVEN_BUCKETS = ("past_due", "due_today", "due_tomorrow")

_BUCKET_LABEL = {
    "past_due": "🔴 Past due",
    "due_today": "🟠 Due today",
    "due_tomorrow": "🟡 Due tomorrow",
    "future": "Future",
    "kanban_only": "Buffer",
}


def render_buy_plan():
    st.header("🛒 Buy Plan (Validated)")
    st.caption(
        "Raw materials to buy, driven by open sales orders → BOM → raw materials, "
        "netted against on-hand and open POs. Quantities use the **corrected** "
        "active-recipe BOM (no longer summing archived batch recipes)."
    )

    with st.sidebar:
        st.subheader("⚙️ Buy Plan Settings")
        lookback_months = st.selectbox(
            "Usage lookback (months)",
            options=[12, 24, 36],
            index=1,
            help="History window for the kanban buffer rate (IV30300).",
        )
        include_future = st.checkbox(
            "Include 'future' open orders",
            value=False,
            help="By default only past-due / today / tomorrow orders drive demand.",
        )

    bundle = _run_pipeline(lookback_months, include_future)
    if bundle is None:
        return
    result, purchasable = bundle

    raw_df: pd.DataFrame = result.get("raw_materials", pd.DataFrame())
    missing_bom_df: pd.DataFrame = result.get("missing_bom_report", pd.DataFrame())
    skipped_df: pd.DataFrame = result.get("skipped_report", pd.DataFrame())
    as_of = result.get("as_of", datetime.date.today())

    # Pull the phantom-risk rows (REC- codes / no vendor history) OUT of the
    # trustworthy buy list and into their own review section.
    clean_df, review_df = _partition_review(raw_df, purchasable)
    order_driven, buffer_only = _split(clean_df)

    _render_metrics(order_driven, buffer_only, review_df, missing_bom_df, as_of)
    st.divider()

    # --- The actionable list: raw materials pulled by open orders ---
    st.subheader(f"🔴 Order now — pulled by open orders ({len(order_driven)} items)")
    st.caption(
        "Raw materials where open-order demand exceeds on-hand + open POs. "
        "These are committed within days, so they sit inside any 17-day order horizon."
    )
    if order_driven.empty:
        st.success("No raw materials are short for open orders right now. 🎉")
    else:
        _render_buy_table(order_driven, "buy_plan_order_now.csv")

    # --- Buffer restock: no specific order, just below one month of usage ---
    # Auto-expand when nothing is order-urgent, so the actionable list isn't hidden.
    with st.expander(f"🟡 Buffer restock — below kanban buffer ({len(buffer_only)} items)",
                     expanded=bool(order_driven.empty and not buffer_only.empty)):
        st.caption(
            "Not short for any committed open order — only below ~one month of P80 usage "
            "(high-volume runners like NPK0015 land here). Not date-urgent; size to your buffer policy."
        )
        if buffer_only.empty:
            st.info("All buffers healthy.")
        else:
            _render_buy_table(buffer_only, "buy_plan_buffer.csv")

    st.divider()

    # --- Quarantine: known phantom-buy classes (review, don't auto-order) ---
    _render_review(review_df)

    # --- Transparency: demand that did NOT explode (the silent black holes) ---
    _render_unplanned(missing_bom_df, skipped_df)

    _render_caveats()


# ---------------------------------------------------------------------------

def _run_pipeline(lookback_months: int, include_future: bool):
    """Run the orchestrator and, on the same connection, find which buy-list
    items have any vendor purchase history. Returns (result, purchasable_set)."""
    progress = st.progress(0.0, text="Starting…")
    status = st.empty()

    def stage_cb(i, total, label):
        progress.progress(min(1.0, i / total), text=f"Step {i}/{total}: {label}")
        status.caption(label)

    try:
        with get_pooled_connection() as conn:
            conn.timeout = 60  # abort runaway recursive CTEs
            cur = conn.cursor()
            result = build_integrated_reorder_list(
                cur,
                today=datetime.date.today(),
                lookback_months=lookback_months,
                include_future_demand=include_future,
                stage_cb=stage_cb,
            )
            raw_df = result.get("raw_materials", pd.DataFrame())
            items = [] if raw_df is None or raw_df.empty else raw_df["item_number"].dropna().tolist()
            purchasable = _purchasable_items(cur, items)
    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"Could not build the buy plan: {exc}")
        LOGGER.exception("build_integrated_reorder_list failed")
        return None

    progress.progress(1.0, text="Done ✅")
    progress.empty()
    status.empty()
    return result, purchasable


def _purchasable_items(cursor, items) -> set:
    """Items with at least one historical vendor receipt (POP30310)."""
    if not items:
        return set()
    placeholders = ", ".join("?" for _ in items)
    try:
        cursor.execute(
            f"SELECT DISTINCT RTRIM(ITEMNMBR) FROM POP30310 WHERE ITEMNMBR IN ({placeholders})",
            *items,
        )
        return {r[0].strip() for r in cursor.fetchall()}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("purchasable lookup failed: %s", exc)
        return set(items)  # fail open: don't quarantine if we can't tell


def _review_reason(item: str, purchasable: set) -> str | None:
    name = (item or "").strip()
    if name.upper().startswith("REC-"):
        return "REC- receiving code — stock likely sits under the base code"
    if name not in purchasable:
        return "No vendor purchase history — likely made in-house, not orderable"
    return None


def _partition_review(raw_df: pd.DataFrame, purchasable: set):
    """Split the buy list into (trustworthy, needs-review)."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    reasons = raw_df["item_number"].map(lambda it: _review_reason(it, purchasable))
    review_mask = reasons.notna()
    review_df = raw_df[review_mask].copy()
    if not review_df.empty:
        review_df["review_reason"] = reasons[review_mask].values
    return raw_df[~review_mask].copy(), review_df


def _split(raw_df: pd.DataFrame):
    """Order-driven vs buffer-only, by whether COMMITTED open-order demand alone
    exceeds supply. A tiny past-due order must not relabel a giant kanban-buffer
    shortfall as "order-urgent" (audit finding #13), so we recompute the order-only
    short instead of trusting the urgency bucket."""
    if raw_df is None or raw_df.empty:
        empty = pd.DataFrame()
        return empty, empty
    df = raw_df.copy()
    supply = None
    for c in ("on_hand_main", "on_hand_other", "proxy_credit", "on_order"):
        if c in df.columns:
            supply = df[c].astype(float) if supply is None else supply + df[c].astype(float)
    if "sop_derived_demand" in df.columns:
        sop = df["sop_derived_demand"].astype(float)
        df["order_net"] = (sop - (supply if supply is not None else 0.0)).clip(lower=0.0)
    else:
        df["order_net"] = 0.0
    order_now = df[df["order_net"] > 0].copy()
    buffer = df[df["order_net"] <= 0].copy()
    return order_now, buffer


def _render_metrics(order_driven, buffer_only, review_df, missing_bom_df, as_of):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🔴 Order now", 0 if order_driven.empty else len(order_driven),
                  help="Raw materials short for open orders")
    with c2:
        st.metric("🟡 Buffer restock", 0 if buffer_only.empty else len(buffer_only),
                  help="Below one month of usage, no specific order")
    with c3:
        st.metric("🔎 Needs review", 0 if review_df is None or review_df.empty else len(review_df),
                  help="REC- codes / no vendor history — do not auto-order")
    with c4:
        st.metric("⚠️ Orders w/o BOM", 0 if missing_bom_df is None or missing_bom_df.empty else len(missing_bom_df),
                  help="Open-order demand that did NOT explode — plan manually")
    with c5:
        st.metric("📅 As of", as_of.strftime("%Y-%m-%d"))


def _render_buy_table(df: pd.DataFrame, csv_name: str, extra_first: tuple | None = None):
    # Human-friendly columns in decision order.
    col_map = [
        ("item_number", "Item"),
        ("item_description", "Description"),
        ("net_need", "BUY QTY"),
        ("order_net", "Order-Driven"),
        ("on_hand_main", "On Hand"),
        ("on_order", "On Order"),
        ("driving_parent", "For Product"),
        ("driving_customer", "Customer"),
        ("earliest_req_date", "Needed By"),
        ("driving_bucket", "Demand"),
        ("urgency_label", "Urgency"),
    ]
    if extra_first:
        col_map = [extra_first] + col_map
    present = [(src, lbl) for src, lbl in col_map if src in df.columns]
    out = df[[src for src, _ in present]].copy()
    out.columns = [lbl for _, lbl in present]

    for c in ("BUY QTY", "Order-Driven", "On Hand", "On Order"):
        if c in out.columns:
            out[c] = out[c].astype(float).round(0)
    if "Demand" in out.columns:
        out["Demand"] = out["Demand"].map(_BUCKET_LABEL).fillna(out["Demand"])
    if "BUY QTY" in out.columns:
        out = out.sort_values("BUY QTY", ascending=False)

    st.dataframe(out, use_container_width=True, hide_index=True)
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name=csv_name, mime="text/csv")


def _render_review(review_df: pd.DataFrame):
    n = 0 if review_df is None or review_df.empty else len(review_df)
    with st.expander(f"🔎 Review before ordering — phantom-buy risk ({n})", expanded=n > 0):
        st.caption(
            "These rows are **excluded from the buy list above** because they are a known "
            "false-buy class: a **REC-** receiving code (the real stock is filed under the "
            "base item) or an item with **no vendor purchase history** (made in-house, not "
            "orderable). Confirm the base-code stock / a real vendor before ordering any of these."
        )
        if not n:
            st.success("No phantom-risk rows. 🎉")
            return
        _render_buy_table(
            review_df, "buy_plan_review.csv",
            extra_first=("review_reason", "Why flagged"),
        )


def _render_unplanned(missing_bom_df: pd.DataFrame, skipped_df: pd.DataFrame):
    n_missing = 0 if missing_bom_df is None or missing_bom_df.empty else len(missing_bom_df)
    n_skipped = 0 if skipped_df is None or skipped_df.empty else len(skipped_df)
    header = f"⚠️ Demand that did NOT explode to raw materials ({n_missing + n_skipped})"

    with st.expander(header, expanded=n_missing > 0):
        st.caption(
            "These open-order finished goods produced **no** raw-material demand — "
            "no BOM and no usable production history, or the BOM query was skipped. "
            "Their raws are **missing** from the buy list above. Plan them manually "
            "until a BOM is authored. (This is the silent gap that used to hide, e.g., a "
            "7,000-gal PETRO1300 order.)"
        )
        if n_missing:
            cols = [c for c in ("item_number", "needed_qty", "reason",
                                "driving_parent", "driving_customer") if c in missing_bom_df.columns]
            st.dataframe(missing_bom_df[cols], use_container_width=True, hide_index=True)
        if n_skipped:
            st.markdown("**Skipped (slow/failed BOM query):**")
            cols = [c for c in ("item_number", "qty", "reason", "error",
                                "driving_customer") if c in skipped_df.columns]
            st.dataframe(skipped_df[cols], use_container_width=True, hide_index=True)
        if not n_missing and not n_skipped:
            st.success("Every open-order finished good exploded cleanly. 🎉")


def _render_caveats():
    with st.expander("ℹ️ What this page does and doesn't yet handle", expanded=False):
        st.markdown(
            """
**Fixed and trustworthy now**
- Buy **quantities** use the corrected active-recipe BOM (the 11×–23× over-buy is gone).
- Known **phantom buys** (REC- codes, non-purchasable in-house items) are pulled into the
  *Review* section instead of the main list.
- Demand that has **no BOM** is surfaced instead of silently disappearing.

**Known caveats (being addressed next — see the audit report)**
- **REC- / base codes** are not yet *merged*, so their real net need isn't computed here —
  they're flagged for review rather than netted against base-code stock.
- **Allocations** (`ATYALLOC`) are *not* subtracted from on-hand (doing so risks
  double-counting against open-order demand) — on-hand shown is gross.
- **Packaging / containers** (jugs, boxes, caps) are excluded from this raw-material list.

Full detail: `docs/superpowers/audits/2026-06-05-demand-chain-trust-report.md`.
            """
        )


if __name__ == "__main__":
    render_buy_plan()
