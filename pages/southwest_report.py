"""
Southwest Report — automated regeneration of the raw-material usage workbook
previously maintained by hand (`Southwest Report 2024-2025_BC Edits.xlsx`).

Data source: Microsoft Dynamics GP inventory adjustment history (IV30300),
DOCTYPE=1 IVAD transactions at LOCNCODE='MAIN'. Sign-flipped so consumption
is positive.

Cell-level verified against the original workbook (Jul 2022 NPK11370:
4 IVADJ rows summing to -742.55 → workbook shows 743).
"""
from __future__ import annotations

import io
from datetime import date
from typing import Iterable

import pandas as pd
import pyodbc
import streamlit as st

from secrets_loader import build_connection_string


# ---------- Static reference data ----------

# Product list pulled from the workbook's GP QTY sheet. Stable catalog of
# raw materials tracked on the report.
ITEMS: list[tuple[str, str]] = [
    ("NPK11370", "11-37-0"),
    ("NPKAQUA", "AQUA AMMONIA"),
    ("SO4BORIC", "BORIC ACID"),
    ("SO4BORON", "BORON"),
    ("NO3CA", "CALCIUM NITRATE"),
    ("CHECITRIC", "CITRIC ACID"),
    ("CHEGLUCO", "GLUCOHEPTONATE"),
    ("GRPHUMICLIQ", "HUMIC ACID LIQUID"),
    ("NO3FE", "IRON NITRATE 10%"),
    ("SO4FEDRY19", "IRON SULPHATE (Dry)"),
    ("NPKKOHDRY", "KOH DRY"),
    ("NPKKOHLIQ", "KOH LIQUID 45%"),
    ("NPKKTS", "KTS"),
    ("CHELIGLIQ", "LIGNIN SULFONATE"),
    ("CHELIGDRY", "LIGNIN SULFONATE"),
    ("SO4MG", "MAG SULF ANHYDROUS"),
    ("NO3MG60", "MAG. NITRATE 6.0%"),
    ("NO3MG63", "MAG. NITRATE 6.3%"),
    ("NO3MN", "MANGANESE NITRATE"),
    ("SO4MN32", "MANGANESE SULFATE 32%"),
    ("NPK2800", "N-SURE"),
    ("NPKNPHURCDI", "N-PHURIC"),
    ("NPKPHOS75", "PHOS ACID 75%/85%"),
    ("NPKKCL62", "POTASS.CLORIDE 62"),
    ("NPKKNO3", "POTASS.NITRATE13-045"),
    ("NPKU32", "U-32"),
    ("NPKUREA", "UREA 46%"),
    ("NO3ZN", "ZINC NITRATE"),
    ("SO4ZNSTD", "ZINC SULPHATE 35.5%"),
    ("EDTAFEHE", "IRON HEDTA"),
    ("EDTAACID", "EDTA ACID DRY"),
    ("EDTAZN", "ZINC EDTA 9%"),
    ("SO4CU", "COPPER SULPHATE 25.2%"),
]
ITEM_CODES: list[str] = [c for c, _ in ITEMS]
ITEM_DESC: dict[str, str] = dict(ITEMS)

# Supplier → Dynamics GP VENDORID, per the user's list.
VENDORS: dict[str, str] = {
    "Hawkins Inc.": "HAWKINS",
    "Nutrien AG Solutions": "CROPPROD",
    "Helm Fertilizer Corp": "HELM",
    "GreenView Chemical": "GREENVIEW",
    "SQM North American": "SQMNORTH",
    "Borregaard USA": "LIGNOTEC",
    "Frit Industries": "FRIT",
    "US Chemland": "CHEMLAND",
    "Valudor Products": "VALUDORPRODUCTS",
    "Arclin": "ARCLINUSA",
    "IMC Agribusiness": "IMCAGRIB",
}

LOCATION = "MAIN"


# ---------- Fiscal calendar helpers ----------

def fiscal_year(d: pd.Timestamp | date) -> int:
    """FY ending June 30. Jul 2022 → FY2023."""
    ts = pd.Timestamp(d)
    return ts.year + 1 if ts.month >= 7 else ts.year


def fiscal_quarter(d: pd.Timestamp | date) -> int:
    """Jul-Sep=1, Oct-Dec=2, Jan-Mar=3, Apr-Jun=4."""
    month = pd.Timestamp(d).month
    return {7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 12: 2,
            1: 3, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4}[month]


# ---------- Data loaders ----------

def _connect() -> pyodbc.Connection:
    conn_str, _, _, _ = build_connection_string()
    return pyodbc.connect(conn_str)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_monthly_usage(start: date, end: date) -> pd.DataFrame:
    """
    Return DataFrame(item, month_end, qty) — monthly consumption per item,
    summed from IV30300 IVAD adjustments at MAIN. Qty is sign-flipped so
    consumption is positive.
    """
    # Also match REC- variants (REC-NO3FE is just the receiving code for NO3FE).
    all_codes = ITEM_CODES + [f"REC-{c}" for c in ITEM_CODES]
    placeholders = ", ".join("?" for _ in all_codes)
    # Consumption only: sum the magnitudes of negative adjustment transactions.
    # Positive IVAD transactions are inventory count corrections / write-ups,
    # which the original workbook excluded (verified against Sep 2022 NPK11370:
    # workbook=272 matches |negative-only|, not net, which had an +87k count fix).
    # Strip REC- prefix in SQL so REC-NO3FE rolls up under NO3FE.
    query = f"""
        SELECT REPLACE(RTRIM(ITEMNMBR), 'REC-', '') AS item,
               DATEFROMPARTS(YEAR(DOCDATE), MONTH(DOCDATE), 1) AS month_start,
               -SUM(CASE WHEN TRXQTY < 0 THEN TRXQTY ELSE 0 END) AS qty
        FROM IV30300
        WHERE DOCTYPE = 1
          AND TRXSORCE LIKE 'IVAD%'
          AND RTRIM(TRXLOCTN) = ?
          AND DOCDATE BETWEEN ? AND ?
          AND RTRIM(ITEMNMBR) IN ({placeholders})
        GROUP BY REPLACE(RTRIM(ITEMNMBR), 'REC-', ''), YEAR(DOCDATE), MONTH(DOCDATE)
    """
    params = [LOCATION, start, end, *all_codes]
    with _connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return df
    df["month_start"] = pd.to_datetime(df["month_start"])
    df["qty"] = df["qty"].astype(float)
    df["fy"] = df["month_start"].apply(fiscal_year)
    df["fq"] = df["month_start"].apply(fiscal_quarter)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_on_hand(snapshot_date: date) -> pd.DataFrame:
    """
    On-hand as of `snapshot_date`. Since IV00102 is a live table (no history),
    we approximate any past date by taking current on-hand and rolling back
    transactions after the snapshot. For today/future dates, just read IV00102.
    """
    placeholders = ", ".join("?" for _ in ITEM_CODES)
    today = date.today()

    with _connect() as conn:
        live = pd.read_sql(
            f"""
            SELECT RTRIM(ITEMNMBR) AS item, SUM(QTYONHND) AS qty
            FROM IV00102
            WHERE ITEMNMBR IN ({placeholders}) AND LOCNCODE = ?
            GROUP BY ITEMNMBR
            """,
            conn,
            params=[*ITEM_CODES, LOCATION],
        )
        if snapshot_date < today:
            # Reverse all transactions after snapshot_date: current_qty - sum(trx_after)
            adjust = pd.read_sql(
                f"""
                SELECT RTRIM(ITEMNMBR) AS item, SUM(TRXQTY) AS delta
                FROM IV30300
                WHERE RTRIM(TRXLOCTN) = ?
                  AND DOCDATE > ?
                  AND RTRIM(ITEMNMBR) IN ({placeholders})
                GROUP BY ITEMNMBR
                """,
                conn,
                params=[LOCATION, snapshot_date, *ITEM_CODES],
            )
        else:
            adjust = pd.DataFrame(columns=["item", "delta"])

    on_hand = live.set_index("item")["qty"].astype(float)
    if not adjust.empty:
        delta = adjust.set_index("item")["delta"].astype(float)
        on_hand = on_hand.subtract(delta, fill_value=0.0)

    return (
        on_hand.reindex(ITEM_CODES, fill_value=0.0)
        .rename("qty")
        .reset_index()
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_primary_vendor_map(vendor_ids: tuple[str, ...]) -> dict[str, str]:
    """
    Map each tracked item to its primary receipt vendor (last 3 years),
    restricted to the given vendor set. Returns {item: VENDORID}.
    """
    if not vendor_ids:
        return {}
    item_ph = ", ".join("?" for _ in ITEM_CODES)
    vendor_ph = ", ".join("?" for _ in vendor_ids)
    query = f"""
        WITH receipts AS (
            SELECT RTRIM(l.ITEMNMBR) AS item,
                   RTRIM(h.VENDORID) AS vendor,
                   SUM(l.UMQTYINB) AS qty
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.POPTYPE NOT IN (4, 5)
              AND h.VOIDSTTS = 0
              AND h.RECEIPTDATE >= DATEADD(year, -3, GETDATE())
              AND RTRIM(l.ITEMNMBR) IN ({item_ph})
              AND RTRIM(h.VENDORID) IN ({vendor_ph})
            GROUP BY l.ITEMNMBR, h.VENDORID
        ),
        ranked AS (
            SELECT item, vendor, qty,
                   ROW_NUMBER() OVER (PARTITION BY item ORDER BY qty DESC) AS rnk
            FROM receipts
        )
        SELECT item, vendor FROM ranked WHERE rnk = 1
    """
    with _connect() as conn:
        rows = pd.read_sql(query, conn, params=[*ITEM_CODES, *vendor_ids])
    return dict(zip(rows["item"], rows["vendor"]))


# ---------- Pivoting ----------

def pivot_yearly(df: pd.DataFrame, items: Iterable[str]) -> pd.DataFrame:
    items = list(items)
    grid = (
        df.groupby(["item", "fy"])["qty"]
        .sum()
        .unstack("fy", fill_value=0.0)
        .reindex(items, fill_value=0.0)
        .sort_index(axis=1)
    )
    grid.insert(0, "Description", [ITEM_DESC.get(i, "") for i in grid.index])
    return grid


def pivot_quarterly(df: pd.DataFrame, items: Iterable[str]) -> pd.DataFrame:
    items = list(items)
    fy_q = (
        df.assign(label=lambda d: "FY" + d["fy"].astype(str) + "-Q" + d["fq"].astype(str))
        .groupby(["item", "label"])["qty"]
        .sum()
        .unstack("label", fill_value=0.0)
        .reindex(items, fill_value=0.0)
        .sort_index(axis=1)
    )
    fy_q.insert(0, "Description", [ITEM_DESC.get(i, "") for i in fy_q.index])
    return fy_q


# ---------- Excel export ----------

def build_excel(yearly: pd.DataFrame, quarterly: pd.DataFrame, on_hand: pd.DataFrame, snapshot_date: date) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        yearly.to_excel(writer, sheet_name="New Yearly Report")
        quarterly.to_excel(writer, sheet_name="New Quarterly Report")
        on_hand_out = on_hand.copy()
        on_hand_out.columns = ["Item #", f"GP QTY on Hand as of {snapshot_date.strftime('%m/%d/%Y')}"]
        on_hand_out.insert(1, "Product", [ITEM_DESC.get(c, "") for c in on_hand_out["Item #"]])
        on_hand_out["U of M"] = "LBS"
        on_hand_out.to_excel(writer, sheet_name="GP QTY", index=False)
    buf.seek(0)
    return buf.getvalue()


# ---------- Page ----------

def render_page() -> None:
    st.title("Southwest Report — Raw Material Usage")
    st.caption(
        "Automated from Dynamics GP (`IV30300` IVAD consumption at MAIN). "
        "Replaces the manually-maintained `Southwest Report *_BC Edits.xlsx`."
    )

    # --- Sidebar filters ---
    st.sidebar.header("Report Parameters")
    today = date.today()
    current_fy = fiscal_year(today)

    fy_start = st.sidebar.number_input(
        "Start fiscal year", min_value=2014, max_value=current_fy, value=max(2014, current_fy - 9),
    )
    fy_end = st.sidebar.number_input(
        "End fiscal year", min_value=2014, max_value=current_fy, value=current_fy,
    )
    if fy_end < fy_start:
        st.sidebar.error("End FY must be ≥ Start FY")
        return

    vendor_choices = ["All vendors"] + list(VENDORS.keys())
    selected_vendor_label = st.sidebar.selectbox("Filter items by primary vendor", vendor_choices)
    selected_vendor_id = VENDORS.get(selected_vendor_label) if selected_vendor_label != "All vendors" else None

    snapshot_date = st.sidebar.date_input("On-hand snapshot date", value=today)

    # --- Data fetch ---
    start_date = date(int(fy_start) - 1, 7, 1)
    end_date = date(int(fy_end), 6, 30)

    with st.spinner("Pulling consumption history from GP..."):
        usage = fetch_monthly_usage(start_date, end_date)

    if usage.empty:
        st.warning("No consumption records found for the selected range.")
        return

    # Filter item set by primary vendor if requested
    items_for_report = ITEM_CODES
    if selected_vendor_id:
        vmap = fetch_primary_vendor_map(tuple(VENDORS.values()))
        items_for_report = [i for i in ITEM_CODES if vmap.get(i) == selected_vendor_id]
        if not items_for_report:
            st.info(
                f"No tracked items have **{selected_vendor_label}** as their primary PO vendor "
                f"in the last 3 years. Showing all items."
            )
            items_for_report = ITEM_CODES

    # --- KPIs ---
    filtered = usage[usage["item"].isin(items_for_report)]
    total_lbs = filtered["qty"].sum()
    latest_fy = filtered["fy"].max() if not filtered.empty else None
    latest_fy_total = filtered.loc[filtered["fy"] == latest_fy, "qty"].sum() if latest_fy else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Items tracked", f"{len(items_for_report)}")
    k2.metric(f"Total consumption FY{fy_start}–FY{fy_end}", f"{total_lbs:,.0f} LBS")
    k3.metric(f"FY{latest_fy} consumption" if latest_fy else "FY —", f"{latest_fy_total:,.0f} LBS")

    # --- Yearly table ---
    st.subheader(f"Yearly Product Usage (FY{fy_start}–FY{fy_end})")
    yearly = pivot_yearly(filtered, items_for_report)
    yearly.columns = ["Description"] + [f"FY{int(c)}" for c in yearly.columns if c != "Description"]
    st.dataframe(yearly.style.format("{:,.0f}", subset=[c for c in yearly.columns if c != "Description"]))

    # --- Quarterly table ---
    st.subheader("Quarterly Product Usage")
    quarterly = pivot_quarterly(filtered, items_for_report)
    st.dataframe(quarterly.style.format("{:,.0f}", subset=[c for c in quarterly.columns if c != "Description"]))

    # --- On-hand ---
    st.subheader(f"GP On-Hand as of {snapshot_date:%Y-%m-%d}")
    on_hand = fetch_on_hand(snapshot_date)
    on_hand_display = on_hand[on_hand["item"].isin(items_for_report)].copy()
    on_hand_display["Product"] = on_hand_display["item"].map(ITEM_DESC)
    on_hand_display = on_hand_display[["item", "Product", "qty"]].rename(columns={"item": "Item #", "qty": "LBS On Hand"})
    st.dataframe(on_hand_display.style.format({"LBS On Hand": "{:,.0f}"}))

    # --- Excel export ---
    excel_bytes = build_excel(yearly, quarterly, on_hand_display[["Item #", "LBS On Hand"]].rename(columns={"LBS On Hand": "Qty"}), snapshot_date)
    vendor_suffix = f"_{selected_vendor_id}" if selected_vendor_id else ""
    st.download_button(
        "📥 Download Excel (matching workbook layout)",
        data=excel_bytes,
        file_name=f"Southwest_Report_FY{fy_start}-FY{fy_end}{vendor_suffix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("How this report is generated"):
        st.markdown(
            """
            **Source**: `IV30300` inventory transaction history, filtered to:

            - `DOCTYPE = 1` (Inventory Adjustment)
            - `TRXSORCE LIKE 'IVAD%'` (IV Adjustment module — excludes PO receipts, sales, transfers)
            - `TRXLOCTN = 'MAIN'` (Chemical Dynamics Micro Plant)
            - `DOCDATE` within the selected fiscal range
            - Only negative `TRXQTY` (consumption); positive adjustments are
              inventory count corrections and are excluded

            Quantities are sign-flipped so consumption shows as positive. Grouped by
            `ITEMNMBR` × `YEAR(DOCDATE)` × `MONTH(DOCDATE)`, then rolled to fiscal year
            and fiscal quarter (Jul–Jun fiscal calendar).

            **Cell-verified** against the original workbook: Jul 2022 `NPK11370` IVADJ
            records sum to 742.55 → workbook shows 743.

            **On-hand** comes from `IV00102.QTYONHND` at `LOCNCODE = 'MAIN'`. Past
            dates are approximated by rolling back post-snapshot transactions.
            """
        )


render_page()
