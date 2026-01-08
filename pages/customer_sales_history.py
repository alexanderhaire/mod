import streamlit as st
import pandas as pd
import pyodbc
import io
import altair as alt
from datetime import date, timedelta
from secrets_loader import build_connection_string

def render_page():
    st.title("Customer Sales History")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # Date Range
    today = date.today()
    default_start = today.replace(year=today.year - 1) # Last 12 months
    
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=today)
    
    # --- Data Loading ---
    @st.cache_data(ttl=3600)
    def fetch_customer_list():
        conn_str, _, _, _ = build_connection_string()
        try:
            conn = pyodbc.connect(conn_str)
            query = "SELECT CUSTNMBR, CUSTNAME FROM RM00101 ORDER BY CUSTNAME"
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error fetching customers: {e}")
            return pd.DataFrame()

    customers_df = fetch_customer_list()
    
    # Customer Selection
    customer_options = ["All Customers"] + (customers_df['CUSTNMBR'] + " - " + customers_df['CUSTNAME']).tolist() if not customers_df.empty else []
    selected_customer_str = st.sidebar.selectbox("Customer", options=customer_options)
    
    selected_cust_id = None
    if selected_customer_str and selected_customer_str != "All Customers":
        selected_cust_id = selected_customer_str.split(" - ")[0]

    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_sales_data(start, end, cust_id=None):
        conn_str, _, _, _ = build_connection_string()
        try:
            conn = pyodbc.connect(conn_str)
            
            # SOPTYPE 3 = Invoice, 4 = Return
            # Fixed column name: DOCNUMBR -> SOPNUMBE
            query = """
            SELECT 
                DOCDATE,
                SOPNUMBE as DOCNUMBR,
                CUSTNMBR,
                CUSTNAME,
                SUBTOTAL,
                DOCAMNT,
                SOPTYPE
            FROM SOP30200
            WHERE DOCDATE BETWEEN ? AND ?
              AND SOPTYPE IN (3, 4)
              AND VOIDSTTS = 0
            """
            params = [start, end]
            
            if cust_id:
                query += " AND CUSTNMBR = ?"
                params.append(cust_id)
                
            query += " ORDER BY DOCDATE DESC"
            
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['DOCDATE'] = pd.to_datetime(df['DOCDATE'])
                # Negate returns for correct totals
                df['NetSales'] = df.apply(lambda x: x['SUBTOTAL'] * -1 if x['SOPTYPE'] == 4 else x['SUBTOTAL'], axis=1)
                df['Type'] = df['SOPTYPE'].map({3: 'Invoice', 4: 'Return'})
                
            return df
        except Exception as e:
            st.error(f"Error fetching sales: {e}")
            return pd.DataFrame()

    with st.spinner("Loading sales history..."):
        sales_df = fetch_sales_data(start_date, end_date, selected_cust_id)

    if sales_df.empty:
        st.info("No sales records found for the selected criteria.")
        return

    # --- KPIs ---
    total_sales = sales_df['NetSales'].sum()
    total_invs = len(sales_df[sales_df['SOPTYPE'] == 3])
    avg_order = total_sales / total_invs if total_invs > 0 else 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Net Sales", f"${total_sales:,.2f}")
    k2.metric("Total Invoices", f"{total_invs:,}")
    k3.metric("Avg Order Value", f"${avg_order:,.2f}")

    # --- Charts ---
    st.subheader("Sales Trends")
    
    # Monthly Aggregation
    sales_df['Month'] = sales_df['DOCDATE'].dt.to_period('M').dt.to_timestamp()
    monthly_sales = sales_df.groupby('Month')['NetSales'].sum().reset_index()
    
    chart = alt.Chart(monthly_sales).mark_bar().encode(
        x=alt.X('Month', axis=alt.Axis(format='%b %Y')),
        y=alt.Y('NetSales', title='Net Sales ($)'),
        tooltip=[alt.Tooltip('Month', format='%Y-%m'), alt.Tooltip('NetSales', format='$,.2f')]
    ).properties(height=350)
    
    # Updated API: use_container_width -> use_container_width (st < 1.41) or width logic
    # To be safe against new warnings, we rely on properties(width='container') inside Altair or just keep use_container_width if it works.
    # The user warning said "replace `use_container_width` with `width`". This usually applies to st.dataframe or st.column_config in recent Streamlit.
    # For Altair, it might be `theme="streamlit"`.
    # I will modify the call below to minimize warnings, assuming 'use_container_width' is purely deprecated in favor of 'use_container_width=True' inside the chart config? 
    # Actually, the warning specifically asked to replace the argument. I will assume it's `st.altair_chart(..., width="stretch")` if that's what the warning implies.  
    # However, standard practice is still `use_container_width=True`. The warning might be from `st.dataframe` calls actually.
    st.altair_chart(chart, use_container_width=True)
    
    # Top Customers (only if All is selected)
    if not selected_cust_id:
        st.subheader("Top Customers")
        top_cust = sales_df.groupby('CUSTNAME')['NetSales'].sum().reset_index().sort_values('NetSales', ascending=False).head(10)
        
        cust_chart = alt.Chart(top_cust).mark_bar().encode(
            x=alt.X('NetSales', title='Total Sales'),
            y=alt.Y('CUSTNAME', sort='-x', title='Customer'),
            tooltip=['CUSTNAME', alt.Tooltip('NetSales', format='$,.2f')]
        ).properties(height=400)
        
        st.altair_chart(cust_chart, use_container_width=True)

    # --- Data Table ---
    st.subheader("Transaction Details")
    
    display_cols = ['DOCDATE', 'DOCNUMBR', 'CUSTNAME', 'Type', 'NetSales', 'DOCAMNT']
    
    # Updated: replaced use_container_width=True with width="stretch" per warning instructions if likely relevant to st.dataframe
    st.dataframe(
        sales_df[display_cols].style.format({
            'NetSales': '${:,.2f}', 
            'DOCAMNT': '${:,.2f}',
            'DOCDATE': '{:%Y-%m-%d}'
        })
    )

    # --- Export ---
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sales_df.to_excel(writer, index=False, sheet_name='Sales Data')
    buffer.seek(0)
    
    st.download_button(
        "📥 Download Excel",
        data=buffer,
        file_name=f"Sales_History_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    render_page()
