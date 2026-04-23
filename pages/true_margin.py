"""
True Margin Over Time — BOM-theoretical finished-goods margin tracker.

For each invoice line of a finished good, explode the manufacturing BOM and
look up each raw material's most-recent PO receipt cost AS OF the invoice
date. True margin = sale price - sum(design_qty * rm_cost_at_date).

Data sources (Dynamics GP):
  - Sales   : SOP30300 lines + SOP30200 header (SOPTYPE 3=Invoice, 4=Return)
  - BOM     : BM010115 via inventory_queries.fetch_recursive_bom_for_item
  - RM cost : POP30310 receipt lines + POP30300 receipt header (UNITCOST,
              UMQTYINB, RECEIPTDATE) — returns/invoice-matches excluded
  - Fallback: IV00101.CURRCOST when no prior receipt exists for a component
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
    alt.data_transformers.disable_max_rows()
except ImportError:
    alt = None

from constants import RAW_MATERIAL_CLASS_CODES
from db_pool import get_connection
from inventory_queries import fetch_recursive_bom_for_item


st.set_page_config(page_title="True Margin Over Time", layout="wide")
st.title("True Margin Over Time")
st.caption(
    "True margin per finished good over time. Default cost basis is the actual "
    "manufacturing-order cost recorded in IV30300 (same cost layer GP uses for "
    "COGS); switchable to BOM-theoretical (explode BM010115, price each RM at "
    "its latest PO receipt as of the sale date)."
)


# ---------- Finished-good picker population ----------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_finished_goods_with_sales(lookback_days: int = 730) -> pd.DataFrame:
    """Items that (a) aren't in the raw-material class list, and (b) have shipped
    sales in the last `lookback_days` days. Returns Revenue, GpCogs, Margin, and
    MarginPct computed from SOP30300 line totals (GpCogs = sale-line EXTDCOST,
    which GP pulls from the IV30300 cost layer — same basis as the Actual-MO
    cost path in this page). Sorted by MarginPct ASC (worst first)."""
    rm_list = "', '".join(RAW_MATERIAL_CLASS_CODES)
    rm_filter = f"UPPER(LTRIM(RTRIM(i.ITMCLSCD))) NOT IN ('{rm_list}')"
    query = f"""
        SELECT
            RTRIM(l.ITEMNMBR) AS ITEMNMBR,
            MAX(RTRIM(i.ITEMDESC)) AS ITEMDESC,
            MAX(RTRIM(i.ITMCLSCD)) AS ITMCLSCD,
            SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.XTNDPRCE) ELSE ABS(l.XTNDPRCE) END) AS Revenue,
            SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.EXTDCOST) ELSE ABS(l.EXTDCOST) END) AS GpCogs,
            SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS Qty
        FROM SOP30300 l
        JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        JOIN IV00101 i ON i.ITEMNMBR = l.ITEMNMBR
        WHERE h.DOCDATE >= DATEADD(day, -?, CAST(GETDATE() AS DATE))
          AND l.SOPTYPE IN (3, 4)
          AND {rm_filter}
        GROUP BY RTRIM(l.ITEMNMBR)
        HAVING SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) > 0
           AND SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.XTNDPRCE) ELSE ABS(l.XTNDPRCE) END) > 0
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=[lookback_days])
    if df.empty:
        return df
    df["Revenue"] = df["Revenue"].astype(float)
    df["GpCogs"] = df["GpCogs"].astype(float)
    df["Margin"] = df["Revenue"] - df["GpCogs"]
    df["MarginPct"] = df["Margin"] / df["Revenue"] * 100
    return df.sort_values("MarginPct", ascending=True).reset_index(drop=True)


# ---------- Sales (finished good) ----------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fg_sales(item: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Invoice-line detail for a finished good over [start, end]. Returns (returns
    are negative qty/revenue). One row per invoice line."""
    query = """
        SELECT
            h.DOCDATE        AS DocDate,
            RTRIM(l.SOPNUMBE) AS SopNumber,
            l.SOPTYPE        AS SopType,
            RTRIM(h.CUSTNMBR) AS CustNmbr,
            RTRIM(h.CUSTNAME) AS CustName,
            CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END AS Qty,
            CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.XTNDPRCE) ELSE ABS(l.XTNDPRCE) END AS Revenue,
            CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.EXTDCOST) ELSE ABS(l.EXTDCOST) END AS GpCogs
        FROM SOP30300 l
        JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        WHERE l.ITEMNMBR = ?
          AND l.SOPTYPE IN (3, 4)
          AND h.DOCDATE BETWEEN ? AND ?
          AND ABS(l.QUANTITY) > 0
        ORDER BY h.DOCDATE, l.SOPNUMBE
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=[item, start, end])
    if df.empty:
        return df
    df["DocDate"] = pd.to_datetime(df["DocDate"]).dt.normalize()
    for col in ("Qty", "Revenue", "GpCogs"):
        df[col] = df[col].astype(float)
    df["UnitPrice"] = df["Revenue"] / df["Qty"].replace(0, pd.NA)
    return df


# ---------- BOM explosion (with 00-parent fallback) ----------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bom(item: str) -> pd.DataFrame:
    """Return DataFrame(RawMaterial, Design_Qty) — fully exploded to leaf RMs.
    Falls back to the '00' parent variant (e.g. SOARBLM02 -> SOARBLM00) if the
    exact item has no BOM rows, per the project's BOM normalization note."""
    def _explode(parent: str) -> pd.DataFrame:
        with get_connection() as conn:
            cursor = conn.cursor()
            rows, _ = fetch_recursive_bom_for_item(cursor, parent)
        if not rows:
            return pd.DataFrame(columns=["RawMaterial", "Design_Qty"])
        return pd.DataFrame(
            [(str(r[0]).strip(), float(r[1])) for r in rows],
            columns=["RawMaterial", "Design_Qty"],
        )

    df = _explode(item)
    if df.empty and len(item) >= 2 and item[-2:].isdigit():
        df = _explode(item[:-2] + "00")
    return df


# ---------- Raw material cost history (for cost-as-of-date lookup) ----------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rm_cost_history(items: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Per-receipt unit-cost history for the given RMs, normalized to BASE unit
    (UNITCOST / UMQTYINB). Also pulls receipts from BEFORE `start` so that a
    sale at `start` can look back to the last receipt. Exclusions mirror the
    project's canonical receipt-history query (see market_insights.py)."""
    if not items:
        return pd.DataFrame(columns=["RawMaterial", "ReceiptDate", "UnitCostBase"])

    # Pull receipts from 2 years before the window start to guarantee at least
    # one prior receipt is available for merge_asof lookups.
    lookback_start = dt.date(start.year - 2, start.month, 1)

    placeholders = ",".join("?" for _ in items)
    query = f"""
        SELECT
            RTRIM(l.ITEMNMBR) AS RawMaterial,
            CAST(h.RECEIPTDATE AS DATE) AS ReceiptDate,
            l.UNITCOST AS UnitCost,
            l.UMQTYINB AS UmQtyInBase
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE RTRIM(l.ITEMNMBR) IN ({placeholders})
          AND h.RECEIPTDATE BETWEEN ? AND ?
          AND l.UNITCOST > 0
          AND h.POPTYPE NOT IN (2, 4, 5)
          AND h.VOIDSTTS = 0
          AND l.NONINVEN = 0
          AND NOT EXISTS (
              SELECT 1 FROM POP30310 ret
              JOIN POP30300 reth ON ret.POPRCTNM = reth.POPRCTNM
              WHERE ret.RCPTRETNUM = l.POPRCTNM
                AND reth.POPTYPE IN (4, 5)
          )
        ORDER BY l.ITEMNMBR, h.RECEIPTDATE
    """
    params = [*items, lookback_start, end]
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=["RawMaterial", "ReceiptDate", "UnitCostBase"])
    df["ReceiptDate"] = pd.to_datetime(df["ReceiptDate"]).dt.normalize()
    df["UnitCost"] = df["UnitCost"].astype(float)
    df["UmQtyInBase"] = df["UmQtyInBase"].replace(0, 1).fillna(1).astype(float)
    # Normalize to BASE unit so it matches the BOM Design_Qty units.
    df["UnitCostBase"] = df["UnitCost"] / df["UmQtyInBase"]
    df = df[["RawMaterial", "ReceiptDate", "UnitCostBase"]]
    return df.sort_values(["RawMaterial", "ReceiptDate"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_curr_cost(items: tuple[str, ...]) -> dict[str, float]:
    """IV00101.CURRCOST fallback per item (used when no prior receipt exists)."""
    if not items:
        return {}
    placeholders = ",".join("?" for _ in items)
    query = f"""
        SELECT RTRIM(ITEMNMBR) AS ITEMNMBR, CURRCOST
        FROM IV00101
        WHERE RTRIM(ITEMNMBR) IN ({placeholders})
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=list(items))
    return {row["ITEMNMBR"]: float(row["CURRCOST"] or 0) for _, row in df.iterrows()}


# ---------- Join: cost-as-of-date for each RM on each sale date ----------

def build_cost_as_of_table(
    bom: pd.DataFrame,
    sale_dates: list[pd.Timestamp],
    start: dt.date,
    end: dt.date,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """For each (raw material, sale date) pair, return the latest receipt
    UnitCostBase <= sale date. Falls back to IV00101.CURRCOST if no prior
    receipt exists. Returns (cost_df, fallback_flags_by_rm)."""
    rms = tuple(bom["RawMaterial"].unique())
    cost_hist = fetch_rm_cost_history(rms, start, end)
    curr_cost = fetch_curr_cost(rms)

    sale_dates_df = pd.DataFrame({"SaleDate": pd.to_datetime(sorted(set(sale_dates)))})
    frames: list[pd.DataFrame] = []
    fallback_used: dict[str, bool] = {}

    for rm in rms:
        sub = cost_hist[cost_hist["RawMaterial"] == rm].copy()
        if sub.empty:
            fallback_used[rm] = True
            frame = sale_dates_df.copy()
            frame["RawMaterial"] = rm
            frame["UnitCostBase"] = curr_cost.get(rm, 0.0)
            frames.append(frame[["RawMaterial", "SaleDate", "UnitCostBase"]])
            continue
        sub = sub.sort_values("ReceiptDate")
        merged = pd.merge_asof(
            sale_dates_df.sort_values("SaleDate"),
            sub.rename(columns={"ReceiptDate": "SaleDate"})[["SaleDate", "UnitCostBase"]],
            on="SaleDate",
            direction="backward",
        )
        missing_mask = merged["UnitCostBase"].isna()
        fallback_used[rm] = bool(missing_mask.any())
        if fallback_used[rm]:
            merged.loc[missing_mask, "UnitCostBase"] = curr_cost.get(rm, 0.0)
        merged["RawMaterial"] = rm
        frames.append(merged[["RawMaterial", "SaleDate", "UnitCostBase"]])

    if not frames:
        return pd.DataFrame(columns=["RawMaterial", "SaleDate", "UnitCostBase"]), {}
    return pd.concat(frames, ignore_index=True), fallback_used


def compute_bom_cost_per_unit(bom: pd.DataFrame, cost_as_of: pd.DataFrame) -> pd.DataFrame:
    """For each SaleDate, compute the BOM-theoretical unit cost of the FG =
    sum over components of design_qty * RM unit cost on that date."""
    merged = cost_as_of.merge(bom, on="RawMaterial", how="left")
    merged["ComponentCost"] = merged["Design_Qty"].astype(float) * merged["UnitCostBase"].astype(float)
    per_date = merged.groupby("SaleDate", as_index=False)["ComponentCost"].sum()
    per_date = per_date.rename(columns={"ComponentCost": "BomUnitCost"})
    return per_date


# ---------- Fallback: finished-good own IV30300 receipt-cost history ----------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fg_receipt_cost_history(item: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """For finished goods without a BOM row, use the FG's own inventory
    receipt-cost history from IV30300 (positive TRXQTY rows — manufacturing
    completions / adjustments). Each row is the actual UNITCOST the MO rolled
    up to at that date.

    Pulls 2 years before `start` so sales at the beginning of the window have
    a prior receipt to look back to."""
    lookback_start = dt.date(start.year - 2, start.month, 1)
    query = """
        SELECT
            CAST(h.DOCDATE AS DATE) AS ReceiptDate,
            t.TRXQTY               AS Qty,
            t.UNITCOST              AS UnitCost
        FROM IV30300 t
        JOIN IV30200 h
            ON t.DOCNUMBR = h.DOCNUMBR
           AND t.DOCTYPE  = h.IVDOCTYP
        WHERE RTRIM(t.ITEMNMBR) = ?
          AND t.TRXQTY   > 0
          AND t.UNITCOST > 0
          AND h.DOCDATE BETWEEN ? AND ?
        ORDER BY h.DOCDATE
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=[item, lookback_start, end])
    if df.empty:
        return pd.DataFrame(columns=["ReceiptDate", "UnitCost", "Qty"])
    df["ReceiptDate"] = pd.to_datetime(df["ReceiptDate"]).dt.normalize()
    df["UnitCost"] = df["UnitCost"].astype(float)
    df["Qty"] = df["Qty"].astype(float)
    return df.sort_values("ReceiptDate").reset_index(drop=True)


def build_fg_cost_per_date(history: pd.DataFrame, sale_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """For each distinct sale date, return the latest prior FG receipt UnitCost
    (as if you were GP's cost layer). Returns columns (SaleDate, BomUnitCost)
    — reusing `BomUnitCost` as the downstream column name so the rest of the
    pipeline is unchanged."""
    sale_dates_df = pd.DataFrame({"SaleDate": pd.to_datetime(sorted(set(sale_dates)))})
    if history.empty:
        sale_dates_df["BomUnitCost"] = 0.0
        return sale_dates_df
    merged = pd.merge_asof(
        sale_dates_df.sort_values("SaleDate"),
        history.rename(columns={"ReceiptDate": "SaleDate"})[["SaleDate", "UnitCost"]],
        on="SaleDate",
        direction="backward",
    )
    # Sales before the first receipt in history -> use the earliest receipt cost.
    first_cost = float(history["UnitCost"].iloc[0])
    merged["UnitCost"] = merged["UnitCost"].fillna(first_cost)
    return merged.rename(columns={"UnitCost": "BomUnitCost"})[["SaleDate", "BomUnitCost"]]


# ---------- UI ----------

fg_df = fetch_finished_goods_with_sales(lookback_days=730)

if fg_df.empty:
    st.warning("No finished-good sales found in the last 2 years.")
    st.stop()

# Build labels like "SOARBLM02 — Soar BLM 2x  [Margin 12.3% / Rev $123,456]"
# Sorted low → high margin so worst-performing items surface first.
fg_df["Label"] = fg_df.apply(
    lambda r: f"{r['ITEMNMBR']} — {(r['ITEMDESC'] or '').strip()}  "
              f"[Margin {(r['MarginPct'] or 0):.1f}% / Rev ${(r['Revenue'] or 0):,.0f}]",
    axis=1,
)

col_a, col_b, col_c = st.columns([3, 1, 1])
with col_a:
    choice = st.selectbox("Finished good", fg_df["Label"].tolist(), index=0)
    selected_item = fg_df.loc[fg_df["Label"] == choice, "ITEMNMBR"].iloc[0]
with col_b:
    today = dt.date.today()
    default_start = dt.date(today.year - 2, today.month, 1)
    start_date = st.date_input("From", default_start)
with col_c:
    end_date = st.date_input("To", today)

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

col_bucket, col_basis = st.columns([1, 1])
with col_bucket:
    bucket = st.radio(
        "Aggregation",
        ["Monthly", "Quarterly", "Weekly"],
        index=0,
        horizontal=True,
    )
with col_basis:
    basis = st.radio(
        "Cost basis",
        ["Actual MO cost (IV30300)", "BOM-theoretical"],
        index=0,
        horizontal=True,
        help="Actual MO cost = the UNITCOST GP recorded on each manufacturing "
             "completion in IV30300 (matches what GP uses for COGS). "
             "BOM-theoretical = explode BM010115 and price each component at "
             "its latest PO-receipt cost as of the sale date.",
    )

# ---- Fetch core data ----
with st.spinner("Loading sales and cost history..."):
    sales = fetch_fg_sales(selected_item, start_date, end_date)

if sales.empty:
    st.warning(f"No shipped invoices for {selected_item} in the selected range.")
    st.stop()

sale_dates = sales["DocDate"].drop_duplicates().tolist()

fallback_flags: dict[str, bool] = {}
fg_cost_history = pd.DataFrame()
bom = pd.DataFrame()

if basis == "Actual MO cost (IV30300)":
    fg_cost_history = fetch_fg_receipt_cost_history(selected_item, start_date, end_date)
    if fg_cost_history.empty:
        st.warning(
            f"No positive-qty inventory-receipt rows in IV30300 for "
            f"{selected_item} within the lookback window. Falling back to "
            f"BOM-theoretical cost."
        )
        bom = fetch_bom(selected_item)
        if bom.empty:
            st.error(
                f"No IV30300 cost history AND no BOM for {selected_item}. "
                f"Cannot compute a cost basis."
            )
            st.dataframe(sales, use_container_width=True)
            st.stop()
        cost_source = "BOM-theoretical (auto-fallback — no IV30300 history)"
        cost_as_of, fallback_flags = build_cost_as_of_table(bom, sale_dates, start_date, end_date)
        bom_cost_per_date = compute_bom_cost_per_unit(bom, cost_as_of)
    else:
        cost_source = "Actual MO cost (IV30300 — matches GP's recorded COGS)"
        bom_cost_per_date = build_fg_cost_per_date(fg_cost_history, sale_dates)
else:  # BOM-theoretical
    bom = fetch_bom(selected_item)
    if bom.empty:
        st.warning(
            f"No BOM found for {selected_item} (or its '00' parent variant). "
            f"Falling back to Actual MO cost from IV30300."
        )
        fg_cost_history = fetch_fg_receipt_cost_history(selected_item, start_date, end_date)
        if fg_cost_history.empty:
            st.error(
                f"No BOM AND no IV30300 cost history for {selected_item}. "
                f"Cannot compute a cost basis."
            )
            st.dataframe(sales, use_container_width=True)
            st.stop()
        cost_source = "Actual MO cost (IV30300 — auto-fallback, no BOM)"
        bom_cost_per_date = build_fg_cost_per_date(fg_cost_history, sale_dates)
    else:
        cost_source = "BOM-theoretical (BM010115 explosion + RM PO receipts)"
        cost_as_of, fallback_flags = build_cost_as_of_table(bom, sale_dates, start_date, end_date)
        bom_cost_per_date = compute_bom_cost_per_unit(bom, cost_as_of)

st.info(f"Cost source: **{cost_source}**")

# ---- Merge BOM unit cost onto invoice lines ----
sales = sales.merge(
    bom_cost_per_date.rename(columns={"SaleDate": "DocDate"}),
    on="DocDate",
    how="left",
)
sales["BomCogs"] = sales["Qty"] * sales["BomUnitCost"]
sales["TrueMargin"] = sales["Revenue"] - sales["BomCogs"]
sales["TrueMarginPct"] = sales["TrueMargin"] / sales["Revenue"].replace(0, pd.NA) * 100

# ---- KPIs ----
total_qty = float(sales["Qty"].sum())
total_rev = float(sales["Revenue"].sum())
total_cogs = float(sales["BomCogs"].sum())
total_margin = total_rev - total_cogs
margin_pct = (total_margin / total_rev * 100) if total_rev else 0.0
avg_price = (total_rev / total_qty) if total_qty else 0.0
avg_cogs_unit = (total_cogs / total_qty) if total_qty else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total qty shipped", f"{total_qty:,.0f}")
k2.metric("Revenue", f"${total_rev:,.0f}")
k3.metric("BOM COGS", f"${total_cogs:,.0f}")
k4.metric("True margin", f"${total_margin:,.0f}", f"{margin_pct:.1f}%")
k5.metric("Avg price / unit", f"${avg_price:,.2f}", f"vs ${avg_cogs_unit:,.2f} cost")

# ---- Aggregation ----
bucket_map = {"Weekly": "W-MON", "Monthly": "MS", "Quarterly": "QS"}
freq = bucket_map[bucket]
sales["Bucket"] = sales["DocDate"].dt.to_period(
    {"W-MON": "W", "MS": "M", "QS": "Q"}[freq]
).dt.start_time

agg = sales.groupby("Bucket", as_index=False).agg(
    Qty=("Qty", "sum"),
    Revenue=("Revenue", "sum"),
    BomCogs=("BomCogs", "sum"),
)
agg["AvgPrice"] = agg["Revenue"] / agg["Qty"].replace(0, pd.NA)
agg["AvgBomCost"] = agg["BomCogs"] / agg["Qty"].replace(0, pd.NA)
agg["Margin"] = agg["Revenue"] - agg["BomCogs"]
agg["MarginPct"] = agg["Margin"] / agg["Revenue"].replace(0, pd.NA) * 100

# ---- Chart: price vs cost vs margin ----
if alt is not None and not agg.empty:
    price_cost_df = agg.melt(
        id_vars="Bucket",
        value_vars=["AvgPrice", "AvgBomCost"],
        var_name="Series",
        value_name="Value",
    )
    price_cost_df["Series"] = price_cost_df["Series"].map(
        {"AvgPrice": "Avg sale price", "AvgBomCost": "BOM cost"}
    )
    price_chart = (
        alt.Chart(price_cost_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Bucket:T", title=bucket),
            y=alt.Y("Value:Q", title="$ per unit"),
            color=alt.Color("Series:N", title=None),
            tooltip=[
                alt.Tooltip("Bucket:T", title=bucket),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Value:Q", format="$,.2f"),
            ],
        )
        .properties(height=280, title=f"Unit price vs BOM cost — {selected_item}")
    )

    margin_chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("Bucket:T", title=bucket),
            y=alt.Y("MarginPct:Q", title="Margin %"),
            color=alt.condition(
                alt.datum.MarginPct >= 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
            tooltip=[
                alt.Tooltip("Bucket:T", title=bucket),
                alt.Tooltip("MarginPct:Q", format=".1f", title="Margin %"),
                alt.Tooltip("Margin:Q", format="$,.0f", title="Margin $"),
                alt.Tooltip("Qty:Q", format=",.0f", title="Qty"),
            ],
        )
        .properties(height=220, title="Margin % over time")
    )
    st.altair_chart(price_chart, use_container_width=True)
    st.altair_chart(margin_chart, use_container_width=True)

# ---- Aggregated table ----
st.subheader(f"{bucket} summary")
show_agg = agg.copy()
show_agg["Bucket"] = show_agg["Bucket"].dt.date
st.dataframe(
    show_agg.style.format({
        "Qty": "{:,.0f}",
        "Revenue": "${:,.0f}",
        "BomCogs": "${:,.0f}",
        "AvgPrice": "${:,.2f}",
        "AvgBomCost": "${:,.2f}",
        "Margin": "${:,.0f}",
        "MarginPct": "{:.1f}%",
    }),
    use_container_width=True,
    hide_index=True,
)

# ---- BOM composition OR FG receipt-cost history ----
if not bom.empty:
    with st.expander(f"BOM composition for {selected_item} ({len(bom)} components)"):
        bom_display = bom.copy()
        curr = fetch_curr_cost(tuple(bom_display["RawMaterial"].unique()))
        bom_display["CurrCost"] = bom_display["RawMaterial"].map(curr).fillna(0.0)
        bom_display["ExtCurrCost"] = bom_display["Design_Qty"] * bom_display["CurrCost"]
        bom_display["Fallback?"] = bom_display["RawMaterial"].map(
            lambda rm: "yes" if fallback_flags.get(rm) else ""
        )
        bom_display = bom_display.sort_values("ExtCurrCost", ascending=False)
        st.dataframe(
            bom_display.style.format({
                "Design_Qty": "{:,.4f}",
                "CurrCost": "${:,.4f}",
                "ExtCurrCost": "${:,.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        fallback_rms = [rm for rm, used in fallback_flags.items() if used]
        if fallback_rms:
            st.caption(
                f"Warning: no PO-receipt cost history found for {len(fallback_rms)} "
                f"component(s); used IV00101.CURRCOST as fallback: "
                f"{', '.join(fallback_rms)}"
            )
else:
    with st.expander(
        f"FG receipt-cost history for {selected_item} "
        f"({len(fg_cost_history)} receipts in lookback window)"
    ):
        hist = fg_cost_history.copy()
        hist["ReceiptDate"] = hist["ReceiptDate"].dt.date
        hist = hist.sort_values("ReceiptDate", ascending=False)
        st.dataframe(
            hist.style.format({"Qty": "{:,.2f}", "UnitCost": "${:,.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"No BM010115 BOM exists for {selected_item}. Cost basis is taken "
            f"from IV30300 positive-qty adjustments (manufacturing completions) — "
            f"for each sale, the latest prior receipt's UNITCOST is used."
        )

# ---- Per-invoice drilldown ----
st.subheader("Invoice-line detail")
drill = sales[[
    "DocDate", "SopNumber", "SopType", "CustNmbr", "CustName",
    "Qty", "UnitPrice", "BomUnitCost", "Revenue", "BomCogs",
    "TrueMargin", "TrueMarginPct", "GpCogs",
]].copy()
drill["DocDate"] = drill["DocDate"].dt.date
drill = drill.sort_values("DocDate", ascending=False)
st.dataframe(
    drill.style.format({
        "Qty": "{:,.2f}",
        "UnitPrice": "${:,.2f}",
        "BomUnitCost": "${:,.2f}",
        "Revenue": "${:,.2f}",
        "BomCogs": "${:,.2f}",
        "GpCogs": "${:,.2f}",
        "TrueMargin": "${:,.2f}",
        "TrueMarginPct": "{:.1f}%",
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download invoice detail (CSV)",
    drill.to_csv(index=False).encode("utf-8"),
    file_name=f"true_margin_{selected_item}_{start_date}_{end_date}.csv",
    mime="text/csv",
)
