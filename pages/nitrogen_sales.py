import streamlit as st
import pandas as pd
import pyodbc
import io
import altair as alt
from secrets_loader import build_connection_string

def render_page():
    st.title("Nitrogen Sales Analysis (Q4 2025)")
    st.caption("Analysis of items with 'Nitrogen' flag (Tax Schedule: NIT&TON)")

    # 1. Fetch Data
    @st.cache_data(ttl=3600)
    def fetch_data():
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        
        # Get Items
        item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID = 'NIT&TON'"
        items_df = pd.read_sql(item_query, conn)
        
        if items_df.empty:
            item_query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITMTSHID LIKE '%NIT%'"
            items_df = pd.read_sql(item_query, conn)
            
        if items_df.empty:
            conn.close()
            return pd.DataFrame(), pd.DataFrame(), 0
            
        nitrogen_items = tuple(items_df['ITEMNMBR'].tolist())
        if len(nitrogen_items) == 1:
            nitrogen_items = f"('{nitrogen_items[0]}')"
            
        # Get Sales
        detail_query = f"""
        SELECT 
            h.DOCDATE,
            l.ITEMNMBR,
            MAX(l.ITEMDESC) as Description,
            SUM(l.XTNDPRCE) as TotalSales,
            SUM(l.QUANTITY) as TotalQty
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPNUMBE = l.SOPNUMBE AND h.SOPTYPE = l.SOPTYPE
        WHERE h.DOCDATE BETWEEN '2025-10-01' AND '2025-12-31'
          AND l.ITEMNMBR IN {nitrogen_items}
          AND h.SOPTYPE = 3 
          AND h.VOIDSTTS = 0
        GROUP BY h.DOCDATE, l.ITEMNMBR
        ORDER BY h.DOCDATE
        """
        detail_df = pd.read_sql(detail_query, conn)
        detail_df['DOCDATE'] = pd.to_datetime(detail_df['DOCDATE'])
        
        # Summary
        summary_query = f"""
        SELECT 
            l.ITEMNMBR,
            MAX(l.ITEMDESC) as Description,
            SUM(l.XTNDPRCE) as TotalSales,
            SUM(l.QUANTITY) as TotalQty
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPNUMBE = l.SOPNUMBE AND h.SOPTYPE = l.SOPTYPE
        WHERE h.DOCDATE BETWEEN '2025-10-01' AND '2025-12-31'
          AND l.ITEMNMBR IN {nitrogen_items}
          AND h.SOPTYPE = 3 
          AND h.VOIDSTTS = 0
        GROUP BY l.ITEMNMBR
        ORDER BY TotalSales DESC
        """
        summary_df = pd.read_sql(summary_query, conn)
        
        conn.close()
        return detail_df, summary_df, len(items_df)

    with st.spinner("Loading sales data..."):
        try:
            detail_df, summary_df, total_items_count = fetch_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    if detail_df.empty:
        st.warning("No sales found for Nitrogen items in Q4 2025.")
        return

    # 2. Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${summary_df['TotalSales'].sum():,.2f}")
    col2.metric("Items Sold", len(summary_df))
    col3.metric("Tagged Items (Inventory)", total_items_count)

    # 3. Daily Trend Chart
    st.subheader("Daily Sales Trend")
    daily_sales = detail_df.groupby('DOCDATE')['TotalSales'].sum().reset_index()
    
    chart_daily = alt.Chart(daily_sales).mark_line(point=True).encode(
        x=alt.X('DOCDATE', title='Date'),
        y=alt.Y('TotalSales', title='Sales ($)'),
        tooltip=['DOCDATE', alt.Tooltip('TotalSales', format='$,.2f')]
    ).properties(height=350)
    st.altair_chart(chart_daily, use_container_width=True)

    # 4. Top Items Chart
    st.subheader("Top Selling Items")
    top_n = st.slider("Show Top Items", 5, 50, 15)
    
    chart_items = alt.Chart(summary_df.head(top_n)).mark_bar().encode(
        x=alt.X('ITEMNMBR', sort='-y', title='Item'),
        y=alt.Y('TotalSales', title='Sales ($)'),
        tooltip=['ITEMNMBR', 'Description', alt.Tooltip('TotalSales', format='$,.2f'), 'TotalQty']
    ).properties(height=400)
    st.altair_chart(chart_items, use_container_width=True)

    # 5. Data Preview & Export
    st.subheader("Data Export")
    
    # Generate Excel in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary by Item', index=False)
        detail_df['DOCDATE'] = detail_df['DOCDATE'].dt.date # Format date for Excel
        detail_df.to_excel(writer, sheet_name='Detailed Transactions', index=False)
        
        # Auto-adjust columns width (simple estimation)
        # Using openpyxl
        for sheet_name in writer.sheets:
            sheet = writer.sheets[sheet_name]
            for col_idx in range(1, 6): # Columns A-E
                col_letter = chr(64 + col_idx)
                sheet.column_dimensions[col_letter].width = 20
            
    buffer.seek(0)
    
    st.download_button(
        label="📥 Download Excel Report",
        data=buffer,
        file_name="Nitrogen_Sales_Q4_2025.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
    
    with st.expander("View Raw Data"):
        st.dataframe(detail_df)

if __name__ == "__main__":
    render_page()
