"""
Poly & DCHA85 Forecast — analyzes purchase history, consumption, and
finished goods sales to project needs for the 26/27 fertilizer year.
"""
import streamlit as st
import pandas as pd
import pyodbc
import io
import altair as alt
from datetime import date
from secrets_loader import build_connection_string


def _connect():
    conn_str, _, _, _ = build_connection_string()
    return pyodbc.connect(conn_str)


def fiscal_year(d):
    """FY ending June 30. Jul 2025 → FY2026."""
    ts = pd.Timestamp(d)
    return ts.year + 1 if ts.month >= 7 else ts.year


# --- Fertilizer year boundaries ---
# 25/26: July 1 2025 – June 30 2026
# 26/27: July 1 2026 – June 30 2027
FY_2526_START = date(2025, 7, 1)
FY_2526_END = date(2026, 6, 30)
FY_2627_START = date(2026, 7, 1)
FY_2627_END = date(2027, 6, 30)


def render_page():
    st.title("Poly & DCHA85 — Fertilizer Year Forecast")
    st.caption("Analyzes 25/26 purchase & consumption history to project 26/27 needs.")

    # ------------------------------------------------------------------ #
    # 1. Find the actual GP item numbers for Poly and DCHA85
    # ------------------------------------------------------------------ #
    @st.cache_data(ttl=3600, show_spinner=False)
    def find_items():
        with _connect() as conn:
            df = pd.read_sql("""
                SELECT
                    LTRIM(RTRIM(ITEMNMBR)) AS Item,
                    LTRIM(RTRIM(ITEMDESC)) AS Description,
                    LTRIM(RTRIM(ITMCLSCD)) AS ItemClass
                FROM IV00101
                WHERE ITEMNMBR LIKE '%POLY%'
                   OR ITEMNMBR LIKE '%DCHA%'
                   OR ITEMDESC LIKE '%POLY%'
                   OR ITEMDESC LIKE '%DCHA%'
                   OR ITEMDESC LIKE '%emulsif%'
                ORDER BY ITEMNMBR
            """, conn)
        return df

    items_df = find_items()

    if items_df.empty:
        st.error("Could not find items matching POLY or DCHA85 in IV00101.")
        return

    st.subheader("Matched Items")
    st.dataframe(items_df, hide_index=True, use_container_width=True)

    # Let user confirm which items to analyze
    all_items = items_df["Item"].tolist()
    selected = st.multiselect("Select items to forecast", all_items, default=all_items, key="fc_items")
    if not selected:
        st.info("Select at least one item.")
        return

    placeholders = ", ".join("?" for _ in selected)

    # ------------------------------------------------------------------ #
    # 2. Purchase history (PO receipts)
    # ------------------------------------------------------------------ #
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_po_history(_selected):
        with _connect() as conn:
            ph = ", ".join("?" for _ in _selected)
            # Historical (received/closed POs)
            hist = pd.read_sql(f"""
                SELECT
                    LTRIM(RTRIM(l.ITEMNMBR)) AS Item,
                    LTRIM(RTRIM(h.PONUMBER)) AS PO,
                    LTRIM(RTRIM(h.VENDNAME)) AS Vendor,
                    h.DOCDATE AS PODate,
                    l.QTYINVCD AS QtyReceived,
                    l.UNITCOST,
                    l.EXTDCOST AS TotalCost
                FROM POP30110 l
                JOIN POP30100 h ON l.PONUMBER = h.PONUMBER
                WHERE RTRIM(l.ITEMNMBR) IN ({ph})
                  AND h.VOIDSTTS = 0
                ORDER BY h.DOCDATE DESC
            """, conn, params=list(_selected))

            # Open POs
            open_po = pd.read_sql(f"""
                SELECT
                    LTRIM(RTRIM(l.ITEMNMBR)) AS Item,
                    LTRIM(RTRIM(h.PONUMBER)) AS PO,
                    LTRIM(RTRIM(h.VENDNAME)) AS Vendor,
                    h.DOCDATE AS PODate,
                    (l.QTYORDER - l.QTYCANCE) AS QtyOpen,
                    l.UNITCOST
                FROM POP10110 l
                JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
                WHERE RTRIM(l.ITEMNMBR) IN ({ph})
                  AND h.POSTATUS IN (1, 2, 3)
                ORDER BY h.DOCDATE DESC
            """, conn, params=list(_selected))
        return hist, open_po

    po_hist, po_open = fetch_po_history(tuple(selected))

    # ------------------------------------------------------------------ #
    # 3. Consumption from IV30300 (all outflows: MO picks, adjustments)
    # ------------------------------------------------------------------ #
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_consumption(_selected):
        with _connect() as conn:
            ph = ", ".join("?" for _ in _selected)
            # Also check for REC- variants
            rec_items = [f"REC-{i}" for i in _selected]
            all_items = list(_selected) + rec_items
            ph_all = ", ".join("?" for _ in all_items)

            df = pd.read_sql(f"""
                SELECT
                    REPLACE(LTRIM(RTRIM(ITEMNMBR)), 'REC-', '') AS Item,
                    DOCDATE,
                    TRXQTY,
                    LTRIM(RTRIM(TRXSORCE)) AS Source,
                    DOCTYPE
                FROM IV30300
                WHERE RTRIM(ITEMNMBR) IN ({ph_all})
                  AND TRXQTY < 0
                  AND DOCDATE >= '2023-07-01'
                ORDER BY DOCDATE
            """, conn, params=all_items)
        if not df.empty:
            df["DOCDATE"] = pd.to_datetime(df["DOCDATE"])
            df["Qty"] = df["TRXQTY"].abs()
        return df

    consumption_df = fetch_consumption(tuple(selected))

    # ------------------------------------------------------------------ #
    # 4. Sales of finished goods using these items (via BOM)
    # ------------------------------------------------------------------ #
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_fg_sales(_selected):
        """Finished goods sold that consume these raw materials, via BOM."""
        with _connect() as conn:
            ph = ", ".join("?" for _ in _selected)
            df = pd.read_sql(f"""
                SELECT
                    LTRIM(RTRIM(b.CPN_I)) AS RawItem,
                    LTRIM(RTRIM(b.PPN_I)) AS FinishedGood,
                    b.QUANTITY_I AS QtyPerFG,
                    LTRIM(RTRIM(fg.ITEMDESC)) AS FGDescription,
                    h.DOCDATE,
                    SUM(s.QUANTITY) AS FGQtySold
                FROM BM010115 b
                JOIN SOP30300 s ON s.ITEMNMBR = b.PPN_I
                JOIN SOP30200 h ON s.SOPNUMBE = h.SOPNUMBE AND s.SOPTYPE = h.SOPTYPE
                JOIN IV00101 fg ON b.PPN_I = fg.ITEMNMBR
                WHERE RTRIM(b.CPN_I) IN ({ph})
                  AND h.SOPTYPE = 3
                  AND h.VOIDSTTS = 0
                  AND h.DOCDATE >= '2023-07-01'
                  AND s.QUANTITY > 0
                GROUP BY b.CPN_I, b.PPN_I, b.QUANTITY_I, fg.ITEMDESC, h.DOCDATE
                ORDER BY h.DOCDATE
            """, conn, params=list(_selected))
        if not df.empty:
            df["DOCDATE"] = pd.to_datetime(df["DOCDATE"])
            df["RawMtlConsumed"] = df["FGQtySold"] * df["QtyPerFG"]
        return df

    fg_df = fetch_fg_sales(tuple(selected))

    # ------------------------------------------------------------------ #
    # 5. Current on-hand + on-order
    # ------------------------------------------------------------------ #
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_inventory(_selected):
        with _connect() as conn:
            ph = ", ".join("?" for _ in _selected)
            df = pd.read_sql(f"""
                SELECT
                    LTRIM(RTRIM(ITEMNMBR)) AS Item,
                    SUM(QTYONHND) AS OnHand,
                    SUM(QTYONORD) AS OnOrder
                FROM IV00102
                WHERE RTRIM(ITEMNMBR) IN ({ph})
                GROUP BY ITEMNMBR
            """, conn, params=list(_selected))
        return df

    inv_df = fetch_inventory(tuple(selected))

    # ================================================================== #
    # DISPLAY
    # ================================================================== #

    # --- Current Position ---
    st.subheader("Current Inventory Position")
    if not inv_df.empty:
        st.dataframe(inv_df, hide_index=True, use_container_width=True)
    else:
        st.info("No inventory found.")

    # --- PO History ---
    st.subheader("Purchase Order History")
    if not po_hist.empty:
        po_hist["PODate"] = pd.to_datetime(po_hist["PODate"])
        po_hist["FY"] = po_hist["PODate"].apply(fiscal_year)

        # FY summary
        fy_summary = po_hist.groupby(["Item", "FY"]).agg(
            TotalQty=("QtyReceived", "sum"),
            TotalCost=("TotalCost", "sum"),
            NumPOs=("PO", "nunique"),
            AvgUnitCost=("UNITCOST", "mean"),
        ).reset_index().sort_values(["Item", "FY"])

        st.write("**Purchases by Fertilizer Year**")
        st.dataframe(
            fy_summary.style.format({
                "TotalQty": "{:,.0f}",
                "TotalCost": "${:,.2f}",
                "AvgUnitCost": "${:,.4f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        # Detail
        with st.expander("PO Detail"):
            st.dataframe(
                po_hist.style.format({
                    "QtyReceived": "{:,.0f}",
                    "UNITCOST": "${:,.4f}",
                    "TotalCost": "${:,.2f}",
                    "PODate": "{:%Y-%m-%d}",
                }),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("No PO history found for selected items.")

    # --- Open POs ---
    if not po_open.empty:
        st.write("**Open Purchase Orders**")
        st.dataframe(
            po_open.style.format({
                "QtyOpen": "{:,.0f}",
                "UNITCOST": "${:,.4f}",
                "PODate": "{:%Y-%m-%d}",
            }),
            hide_index=True,
            use_container_width=True,
        )

    # --- Monthly Consumption ---
    st.subheader("Monthly Consumption (IV30300 Outflows)")
    if not consumption_df.empty:
        consumption_df["Month"] = consumption_df["DOCDATE"].dt.to_period("M").dt.to_timestamp()
        consumption_df["FY"] = consumption_df["DOCDATE"].apply(fiscal_year)

        monthly = consumption_df.groupby(["Item", "Month"])["Qty"].sum().reset_index()

        for item in selected:
            item_monthly = monthly[monthly["Item"] == item]
            if item_monthly.empty:
                continue

            st.write(f"**{item}**")
            chart = alt.Chart(item_monthly).mark_bar().encode(
                x=alt.X("Month:T", axis=alt.Axis(format="%b %Y")),
                y=alt.Y("Qty:Q", title="Quantity Consumed"),
                tooltip=[alt.Tooltip("Month:T", format="%Y-%m"), alt.Tooltip("Qty:Q", format=",.0f")],
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

        # FY consumption summary
        fy_cons = consumption_df.groupby(["Item", "FY"])["Qty"].sum().reset_index()
        st.write("**Consumption by Fertilizer Year**")
        st.dataframe(
            fy_cons.style.format({"Qty": "{:,.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No consumption history found.")

    # --- Finished Goods Driving Demand ---
    st.subheader("Finished Goods Consuming These Items (via BOM)")
    if not fg_df.empty:
        fg_df["FY"] = fg_df["DOCDATE"].apply(fiscal_year)

        # Which FGs drive the most consumption?
        fg_summary = fg_df.groupby(["RawItem", "FinishedGood", "FGDescription", "QtyPerFG"]).agg(
            FGQtySold=("FGQtySold", "sum"),
            RawMtlConsumed=("RawMtlConsumed", "sum"),
        ).reset_index().sort_values("RawMtlConsumed", ascending=False)

        st.dataframe(
            fg_summary.style.format({
                "QtyPerFG": "{:,.4f}",
                "FGQtySold": "{:,.0f}",
                "RawMtlConsumed": "{:,.0f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        # FY breakdown of raw material consumed via FG sales
        fg_fy = fg_df.groupby(["RawItem", "FY"])["RawMtlConsumed"].sum().reset_index()
        st.write("**Raw Material Consumed via FG Sales by Fertilizer Year**")
        st.dataframe(
            fg_fy.style.format({"RawMtlConsumed": "{:,.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No BOM linkage found — these items may not be in BM010115, or consumption flows through MO picks only.")

    # ================================================================== #
    # FORECAST SECTION
    # ================================================================== #
    st.divider()
    st.subheader("26/27 Forecast")

    for item in selected:
        st.write(f"### {item}")

        # Consumption data
        if not consumption_df.empty:
            item_cons = consumption_df[consumption_df["Item"] == item]
            fy26 = item_cons[item_cons["FY"] == 2026]["Qty"].sum()  # FY2026 = July 2025 - June 2026

            # Months elapsed in current FY (up to today)
            today = date.today()
            if today >= FY_2526_START:
                months_elapsed = (today.year - 2025) * 12 + today.month - 7 + 1
                months_elapsed = min(months_elapsed, 12)
            else:
                months_elapsed = 0

            if months_elapsed > 0 and months_elapsed < 12:
                annualized = (fy26 / months_elapsed) * 12
                st.write(f"- **25/26 consumption so far**: {fy26:,.0f} ({months_elapsed} months elapsed)")
                st.write(f"- **Annualized 25/26 estimate**: {annualized:,.0f}")
            elif months_elapsed >= 12:
                annualized = fy26
                st.write(f"- **25/26 full-year consumption**: {fy26:,.0f}")
            else:
                annualized = 0
                st.write("- No 25/26 data yet")

            # Prior year comparison
            fy25 = item_cons[item_cons["FY"] == 2025]["Qty"].sum()  # July 2023 - June 2024... wait
            # FY2025 = July 2024 - June 2025
            if fy25 > 0:
                st.write(f"- **24/25 consumption**: {fy25:,.0f}")
                if annualized > 0:
                    yoy = ((annualized - fy25) / fy25) * 100
                    st.write(f"- **Year-over-year trend**: {yoy:+.1f}%")
        else:
            annualized = 0

        # PO data
        if not po_hist.empty:
            item_po = po_hist[po_hist["Item"] == item]
            fy26_po = item_po[item_po["FY"] == 2026]["QtyReceived"].sum()
            fy25_po = item_po[item_po["FY"] == 2025]["QtyReceived"].sum()
            if fy26_po > 0:
                st.write(f"- **25/26 purchased**: {fy26_po:,.0f}")
            if fy25_po > 0:
                st.write(f"- **24/25 purchased**: {fy25_po:,.0f}")

        # Current inventory
        if not inv_df.empty:
            item_inv = inv_df[inv_df["Item"] == item]
            if not item_inv.empty:
                oh = item_inv["OnHand"].iloc[0]
                oo = item_inv["OnOrder"].iloc[0]
                st.write(f"- **Current on-hand**: {oh:,.0f}  |  **On order**: {oo:,.0f}")

        # Projection
        if annualized > 0:
            st.write(f"- **Projected 26/27 need (based on trend)**: {annualized:,.0f}")
        st.write("---")

    # --- Export ---
    st.subheader("Export")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if not po_hist.empty:
            po_hist.to_excel(writer, index=False, sheet_name="PO History")
        if not po_open.empty:
            po_open.to_excel(writer, index=False, sheet_name="Open POs")
        if not consumption_df.empty:
            consumption_df.to_excel(writer, index=False, sheet_name="Consumption")
        if not fg_df.empty:
            fg_df.to_excel(writer, index=False, sheet_name="FG Sales")
        if not inv_df.empty:
            inv_df.to_excel(writer, index=False, sheet_name="Inventory")
    buffer.seek(0)
    st.download_button(
        "Download Full Analysis (Excel)",
        data=buffer,
        file_name="Poly_DCHA85_Forecast_2627.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    render_page()
