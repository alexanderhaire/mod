import streamlit as st
import pandas as pd
import pyodbc
import io
import altair as alt
from datetime import date, timedelta
from secrets_loader import build_connection_string


def render_page():
    st.title("Label Tracking & Forecasting")
    st.caption("Forecast label demand from finished goods sales, track inventory, and identify reorder needs. Vendor: Southern Tape & Label.")

    # --- Sidebar ---
    st.sidebar.header("Settings")
    lookback = st.sidebar.selectbox("Sales Lookback (days)", [30, 60, 90, 180], index=2, key="lbl_lookback")
    safety_days = st.sidebar.slider("Safety Buffer (days)", 3, 30, 7, key="lbl_safety")
    lead_time_default = st.sidebar.slider("Default Lead Time (days)", 7, 60, 21, key="lbl_lead")
    show_all = st.sidebar.checkbox("Show all labels (incl. no demand)", value=False, key="lbl_all")

    conn_str, _, _, _ = build_connection_string()

    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_label_data(_lookback_days):
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # 1. All label items with inventory
        cursor.execute("""
            SELECT
                LTRIM(RTRIM(i.ITEMNMBR))   AS Item,
                LTRIM(RTRIM(i.ITEMDESC))   AS Description,
                CAST(i.CURRCOST AS FLOAT)  AS UnitCost,
                CAST(ISNULL(loc.QTYONHND, 0) AS FLOAT) AS OnHand,
                CAST(ISNULL(loc.QTYONORD, 0) AS FLOAT) AS OnOrder,
                CAST(ISNULL(loc.ORDRPNTQTY, 0) AS FLOAT) AS GPReorderPt
            FROM IV00101 i
            LEFT JOIN IV00102 loc ON i.ITEMNMBR = loc.ITEMNMBR AND loc.LOCNCODE = 'MAIN'
            WHERE i.ITMCLSCD = 'CUSTLABEL'
        """)
        inv_rows = cursor.fetchall()
        inv_cols = [d[0] for d in cursor.description]
        inv_df = pd.DataFrame.from_records(inv_rows, columns=inv_cols)

        # 2. Label demand derived from finished goods sales via BOM
        cursor.execute("""
            SELECT
                LTRIM(RTRIM(b.CPN_I))      AS Item,
                SUM(s.QUANTITY * b.QUANTITY_I) AS LabelsConsumed
            FROM SOP30300 s
            JOIN SOP30200 h ON s.SOPNUMBE = h.SOPNUMBE AND s.SOPTYPE = h.SOPTYPE
            JOIN BM010115 b ON s.ITEMNMBR = b.PPN_I
            JOIN IV00101 lbl ON b.CPN_I = lbl.ITEMNMBR AND lbl.ITMCLSCD = 'CUSTLABEL'
            WHERE h.DOCDATE >= DATEADD(day, -?, GETDATE())
              AND h.SOPTYPE = 3 AND h.VOIDSTTS = 0
              AND s.QUANTITY > 0
            GROUP BY b.CPN_I
        """, [_lookback_days])
        dem_rows = cursor.fetchall()
        dem_cols = [d[0] for d in cursor.description]
        demand_df = pd.DataFrame.from_records(dem_rows, columns=dem_cols)

        # 3. Open POs for labels (by vendor OR item class)
        cursor.execute("""
            SELECT
                LTRIM(RTRIM(l.ITEMNMBR)) AS Item,
                LTRIM(RTRIM(h.PONUMBER)) AS PONumber,
                LTRIM(RTRIM(h.VENDORID)) AS VendorID,
                SUM(l.QTYORDER - l.QTYCANCE) AS POQtyOpen,
                MIN(l.REQDATE) AS EarliestETA
            FROM POP10110 l
            JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
            WHERE h.POSTATUS IN (1, 2, 3)
              AND (
                LTRIM(RTRIM(h.VENDORID)) = 'SOUTHERNTAPE'
                OR EXISTS (
                    SELECT 1 FROM IV00101 i
                    WHERE i.ITEMNMBR = l.ITEMNMBR AND i.ITMCLSCD = 'CUSTLABEL'
                )
              )
            GROUP BY l.ITEMNMBR, h.PONUMBER, h.VENDORID
        """)
        po_rows = cursor.fetchall()
        po_cols = [d[0] for d in cursor.description]
        po_df = pd.DataFrame.from_records(po_rows, columns=po_cols)

        # 4. Top finished goods driving demand for each label (for context)
        cursor.execute("""
            SELECT
                LTRIM(RTRIM(b.CPN_I)) AS Item,
                STRING_AGG(LTRIM(RTRIM(b.PPN_I)), ', ') AS LinkedFGs
            FROM (
                SELECT DISTINCT b2.CPN_I, b2.PPN_I
                FROM BM010115 b2
                JOIN IV00101 lbl ON b2.CPN_I = lbl.ITEMNMBR AND lbl.ITMCLSCD = 'CUSTLABEL'
            ) b
            GROUP BY b.CPN_I
        """)
        fg_rows = cursor.fetchall()
        fg_cols = [d[0] for d in cursor.description]
        fg_df = pd.DataFrame.from_records(fg_rows, columns=fg_cols)

        # 5. Primary vendor per label
        cursor.execute("""
            SELECT
                LTRIM(RTRIM(iv.ITEMNMBR)) AS Item,
                LTRIM(RTRIM(v.VENDNAME))  AS Vendor
            FROM IV00103 iv
            JOIN PM00200 v ON iv.VENDORID = v.VENDORID
            JOIN IV00101 i ON iv.ITEMNMBR = i.ITEMNMBR AND i.ITMCLSCD = 'CUSTLABEL'
        """)
        ven_rows = cursor.fetchall()
        ven_cols = [d[0] for d in cursor.description]
        vendor_df = pd.DataFrame.from_records(ven_rows, columns=ven_cols)
        # Keep only one vendor per item
        vendor_df = vendor_df.drop_duplicates(subset="Item", keep="first")

        conn.close()
        return inv_df, demand_df, po_df, fg_df, vendor_df

    with st.spinner("Loading label data..."):
        inv_df, demand_df, po_df, fg_df, vendor_df = fetch_label_data(lookback)

    if inv_df.empty:
        st.info("No CUSTLABEL items found.")
        return

    # --- Merge everything ---
    df = inv_df.copy()

    if not demand_df.empty:
        demand_df["LabelsConsumed"] = pd.to_numeric(demand_df["LabelsConsumed"], errors="coerce")
        df = df.merge(demand_df, on="Item", how="left")
    else:
        df["LabelsConsumed"] = 0.0

    # Keep raw PO detail for display later
    po_detail_df = po_df.copy() if not po_df.empty else pd.DataFrame()

    if not po_df.empty:
        po_df["POQtyOpen"] = pd.to_numeric(po_df["POQtyOpen"], errors="coerce")
        po_agg = po_df.groupby("Item").agg(
            POQtyOpen=("POQtyOpen", "sum"),
            EarliestETA=("EarliestETA", "min")
        ).reset_index()
        df = df.merge(po_agg, on="Item", how="left")
    else:
        df["POQtyOpen"] = 0.0
        df["EarliestETA"] = None

    if not fg_df.empty:
        df = df.merge(fg_df, on="Item", how="left")
    else:
        df["LinkedFGs"] = ""

    if not vendor_df.empty:
        df = df.merge(vendor_df, on="Item", how="left")
    else:
        df["Vendor"] = ""

    df["LabelsConsumed"] = df["LabelsConsumed"].fillna(0)
    df["POQtyOpen"] = df["POQtyOpen"].fillna(0)
    df["LinkedFGs"] = df["LinkedFGs"].fillna("")
    df["Vendor"] = df["Vendor"].fillna("")

    # --- Calculations ---
    df["AvgDailyDemand"] = (df["LabelsConsumed"] / lookback).round(1)
    df["Available"] = df["OnHand"] + df["OnOrder"] + df["POQtyOpen"]
    df["DaysCoverage"] = df.apply(
        lambda r: round(r["Available"] / r["AvgDailyDemand"], 0) if r["AvgDailyDemand"] > 0 else 999,
        axis=1
    )
    df["ReorderPoint"] = (df["AvgDailyDemand"] * (lead_time_default + safety_days)).round(0)
    df["SuggestedOrderQty"] = (df["ReorderPoint"] * 2 - df["Available"]).clip(lower=0).round(0)

    today = date.today()
    df["MustOrderBy"] = df.apply(
        lambda r: today + timedelta(days=max(0, int(r["DaysCoverage"] - lead_time_default)))
        if r["AvgDailyDemand"] > 0 else None,
        axis=1
    )

    def calc_urgency(r):
        if r["AvgDailyDemand"] == 0:
            return "No Demand"
        if r["DaysCoverage"] <= lead_time_default:
            return "Critical"
        if r["Available"] < r["ReorderPoint"]:
            return "Soon"
        return "OK"

    df["Urgency"] = df.apply(calc_urgency, axis=1)

    # --- Filter ---
    if not show_all:
        df = df[df["AvgDailyDemand"] > 0]

    if df.empty:
        st.info("No labels with demand in the selected period.")
        return

    urgency_filter = st.sidebar.multiselect(
        "Filter by Urgency",
        options=sorted(df["Urgency"].unique()),
        default=[u for u in ["Critical", "Soon", "OK"] if u in df["Urgency"].values],
        key="lbl_urg"
    )
    if urgency_filter:
        df = df[df["Urgency"].isin(urgency_filter)]

    if df.empty:
        st.info("No labels match the selected filters.")
        return

    # --- KPIs ---
    total = len(df)
    critical = len(df[df["Urgency"] == "Critical"])
    soon = len(df[df["Urgency"] == "Soon"])
    ok = len(df[df["Urgency"] == "OK"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Critical", critical)
    k2.metric("Order Soon", soon)
    k3.metric("OK", ok)
    k4.metric("Total Labels", total)

    # --- Main Table ---
    st.subheader("Label Reorder Recommendations")

    display = df[[
        "Item", "Description", "OnHand", "OnOrder", "POQtyOpen",
        "AvgDailyDemand", "DaysCoverage", "ReorderPoint",
        "SuggestedOrderQty", "MustOrderBy", "Urgency", "Vendor", "LinkedFGs"
    ]].sort_values("DaysCoverage")

    def color_urgency(val):
        if val == "Critical":
            return "background-color: #ffcccc"
        if val == "Soon":
            return "background-color: #fff3cd"
        if val == "OK":
            return "background-color: #d4edda"
        return ""

    st.dataframe(
        display,
        column_config={
            "Item": "Label Item",
            "Description": "Description",
            "OnHand": st.column_config.NumberColumn("On Hand", format="%.0f"),
            "OnOrder": st.column_config.NumberColumn("On Order (IV)", format="%.0f"),
            "POQtyOpen": st.column_config.NumberColumn("Open PO Qty", format="%.0f"),
            "AvgDailyDemand": st.column_config.NumberColumn("Daily Demand", format="%.1f"),
            "DaysCoverage": st.column_config.NumberColumn("Days Coverage", format="%.0f"),
            "ReorderPoint": st.column_config.NumberColumn("Reorder Point", format="%.0f"),
            "SuggestedOrderQty": st.column_config.NumberColumn("Suggested Qty", format="%.0f"),
            "MustOrderBy": st.column_config.DateColumn("Must Order By", format="YYYY-MM-DD"),
            "Urgency": "Urgency",
            "Vendor": "Vendor",
            "LinkedFGs": "Linked Finished Goods",
        },
        hide_index=True,
        width='stretch',
    )

    # --- Open POs ---
    if not po_detail_df.empty:
        st.subheader("Open Label Purchase Orders")
        po_display = po_detail_df.copy()
        po_display["POQtyOpen"] = pd.to_numeric(po_display["POQtyOpen"], errors="coerce")
        st.dataframe(
            po_display,
            column_config={
                "Item": "Item",
                "PONumber": "PO #",
                "VendorID": "Vendor",
                "POQtyOpen": st.column_config.NumberColumn("Qty Open", format="%.0f"),
                "EarliestETA": st.column_config.DateColumn("ETA", format="YYYY-MM-DD"),
            },
            hide_index=True,
            width='stretch',
        )
    else:
        st.subheader("Open Label Purchase Orders")
        st.info("No open POs for labels.")

    # --- Charts ---
    has_demand = df[df["AvgDailyDemand"] > 0].copy()

    if not has_demand.empty:
        st.subheader("Top Labels by Daily Demand")
        top = has_demand.nlargest(15, "AvgDailyDemand")
        bar = alt.Chart(top).mark_bar().encode(
            x=alt.X("AvgDailyDemand:Q", title="Avg Daily Demand"),
            y=alt.Y("Item:N", sort="-x", title="Label Item"),
            color=alt.Color("Urgency:N", scale=alt.Scale(
                domain=["Critical", "Soon", "OK", "No Demand"],
                range=["#dc3545", "#ffc107", "#28a745", "#6c757d"]
            )),
            tooltip=["Item", "Description", "AvgDailyDemand", "DaysCoverage", "OnHand"],
        ).properties(height=400)
        st.altair_chart(bar, width='stretch')

        st.subheader("Days of Coverage by Label")
        coverage = has_demand.nsmallest(20, "DaysCoverage")
        cbar = alt.Chart(coverage).mark_bar().encode(
            x=alt.X("DaysCoverage:Q", title="Days of Coverage"),
            y=alt.Y("Item:N", sort="x", title="Label Item"),
            color=alt.Color("Urgency:N", scale=alt.Scale(
                domain=["Critical", "Soon", "OK", "No Demand"],
                range=["#dc3545", "#ffc107", "#28a745", "#6c757d"]
            )),
            tooltip=["Item", "Description", "DaysCoverage", "OnHand", "AvgDailyDemand"],
        ).properties(height=400)
        st.altair_chart(cbar, width='stretch')

    # --- Export ---
    st.subheader("Export")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        display.to_excel(writer, index=False, sheet_name="Label Forecast")
    buffer.seek(0)
    st.download_button(
        "Download Excel",
        data=buffer,
        file_name=f"Label_Forecast_{today}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


render_page()
