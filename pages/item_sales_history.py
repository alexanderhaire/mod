import streamlit as st
import pandas as pd
import pyodbc
import io
import altair as alt
from datetime import date, timedelta
from secrets_loader import build_connection_string

def render_page():
    st.title("Item Sales History")

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Date Range
    today = date.today()
    default_start = today.replace(year=today.year - 1)

    start_date = st.sidebar.date_input("Start Date", value=default_start, key="item_start")
    end_date = st.sidebar.date_input("End Date", value=today, key="item_end")

    # --- Item List ---
    @st.cache_data(ttl=3600)
    def fetch_item_list():
        conn_str, _, _, _ = build_connection_string()
        try:
            conn = pyodbc.connect(conn_str)
            query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 ORDER BY ITEMDESC"
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error fetching items: {e}")
            return pd.DataFrame()

    items_df = fetch_item_list()

    # Item Selection
    item_options = ["All Items"] + (items_df['ITEMNMBR'].str.strip() + " - " + items_df['ITEMDESC'].str.strip()).tolist() if not items_df.empty else []
    selected_item_str = st.sidebar.selectbox("Item", options=item_options)

    selected_item_id = None
    if selected_item_str and selected_item_str != "All Items":
        selected_item_id = selected_item_str.split(" - ")[0].strip()

    # --- Data Loading ---
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_item_sales(start, end, item_id=None):
        conn_str, _, _, _ = build_connection_string()
        try:
            conn = pyodbc.connect(conn_str)

            query = """
            SELECT
                h.DOCDATE,
                h.SOPNUMBE as DOCNUMBR,
                h.CUSTNMBR,
                h.CUSTNAME,
                l.ITEMNMBR,
                l.ITEMDESC,
                l.QUANTITY,
                l.UOFM,
                l.XTNDPRCE,
                l.EXTDCOST,
                h.SOPTYPE
            FROM SOP30200 h
            JOIN SOP30300 l ON h.SOPNUMBE = l.SOPNUMBE AND h.SOPTYPE = l.SOPTYPE
            WHERE h.DOCDATE BETWEEN ? AND ?
              AND h.SOPTYPE IN (3, 4)
              AND h.VOIDSTTS = 0
            """
            params = [start, end]

            if item_id:
                query += " AND l.ITEMNMBR = ?"
                params.append(item_id)

            query += " ORDER BY h.DOCDATE DESC"

            df = pd.read_sql(query, conn, params=params)
            conn.close()

            if not df.empty:
                df['DOCDATE'] = pd.to_datetime(df['DOCDATE'])
                df['NetSales'] = df.apply(lambda x: x['XTNDPRCE'] * -1 if x['SOPTYPE'] == 4 else x['XTNDPRCE'], axis=1)
                df['Type'] = df['SOPTYPE'].map({3: 'Invoice', 4: 'Return'})

            return df
        except Exception as e:
            st.error(f"Error fetching sales: {e}")
            return pd.DataFrame()

    with st.spinner("Loading item sales history..."):
        sales_df = fetch_item_sales(start_date, end_date, selected_item_id)

    if sales_df.empty:
        st.info("No sales records found for the selected criteria.")
        return

    # --- KPIs ---
    total_sales = sales_df['NetSales'].sum()
    unique_items = sales_df['ITEMNMBR'].nunique()
    total_qty = sales_df.loc[sales_df['SOPTYPE'] == 3, 'QUANTITY'].sum() - sales_df.loc[sales_df['SOPTYPE'] == 4, 'QUANTITY'].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Net Sales", f"${total_sales:,.2f}")
    k2.metric("Unique Items Sold", f"{unique_items:,}")
    k3.metric("Total Quantity Sold", f"{total_qty:,.0f}")

    # --- Charts ---
    st.subheader("Sales Trends")

    sales_df['Month'] = sales_df['DOCDATE'].dt.to_period('M').dt.to_timestamp()
    monthly_sales = sales_df.groupby('Month')['NetSales'].sum().reset_index()

    chart = alt.Chart(monthly_sales).mark_bar().encode(
        x=alt.X('Month', axis=alt.Axis(format='%b %Y')),
        y=alt.Y('NetSales', title='Net Sales ($)'),
        tooltip=[alt.Tooltip('Month', format='%Y-%m'), alt.Tooltip('NetSales', format='$,.2f')]
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    # Top Items (only if All Items is selected)
    if not selected_item_id:
        st.subheader("Top Items")
        top_items = sales_df.groupby(['ITEMNMBR', 'ITEMDESC'])['NetSales'].sum().reset_index()
        top_items['Label'] = top_items['ITEMNMBR'].str.strip() + " - " + top_items['ITEMDESC'].str.strip()
        top_items = top_items.sort_values('NetSales', ascending=False).head(10)

        item_chart = alt.Chart(top_items).mark_bar().encode(
            x=alt.X('NetSales', title='Total Sales'),
            y=alt.Y('Label', sort='-x', title='Item'),
            tooltip=['ITEMNMBR', 'ITEMDESC', alt.Tooltip('NetSales', format='$,.2f')]
        ).properties(height=400)

        st.altair_chart(item_chart, use_container_width=True)

    # --- Data Table ---
    st.subheader("Transaction Details")

    display_cols = ['DOCDATE', 'DOCNUMBR', 'CUSTNAME', 'ITEMNMBR', 'ITEMDESC', 'QUANTITY', 'NetSales', 'Type']

    st.dataframe(
        sales_df[display_cols].style.format({
            'NetSales': '${:,.2f}',
            'QUANTITY': '{:,.2f}',
            'DOCDATE': '{:%Y-%m-%d}'
        })
    )

    # --- Export ---
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sales_df.to_excel(writer, index=False, sheet_name='Item Sales Data')
    buffer.seek(0)

    st.download_button(
        "📥 Download Excel",
        data=buffer,
        file_name=f"Item_Sales_History_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    render_page()
