import streamlit as st
import pandas as pd
import pyodbc
import io
from datetime import date
from secrets_loader import build_connection_string

def render_page():
    st.title("Perpetual Inventory Report")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        target_date = st.date_input("Inventory As Of Date", value=date(2025, 9, 30))
    
    st.caption(f"Showing inventory quantities for **MAIN** location, calculated as of {target_date}. Valued at *Current Cost*.")

    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_inventory_data(as_of_date):
        conn_str, _, _, _ = build_connection_string()
        try:
            conn = pyodbc.connect(conn_str)
            
            # 1. Get Base Item Info + Current Qty (MAIN Location Only)
            query_base = """
            SELECT 
                T1.ITEMNMBR,
                T1.ITEMDESC,
                T1.CURRCOST,
                T2.QTYONHND as CurrentQty,
                T3.ACTNUMST as GLCode
            FROM IV00101 T1
            JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
            LEFT JOIN GL00105 T3 ON T1.IVIVINDX = T3.ACTINDX
            WHERE T2.LOCNCODE = 'MAIN'
            """
            base_df = pd.read_sql(query_base, conn)
            
            # 2. Get Inventory Changes AFTER target date (MAIN Location Only)
            query_history = """
            SELECT 
                ITEMNMBR,
                SUM(TRXQTY) as QtyChange
            FROM IV30300
            WHERE DOCDATE > ? 
              AND TRXLOCTN = 'MAIN'
            GROUP BY ITEMNMBR
            """
            
            # 3. Get Last Cost AS OF target date (Approximation via Last Receipt/Adj)
            # We look for the receipts (DOCTYPE=4) or Adjustments (DOCTYPE=1) 
            # We want the *latest* one on or before the target date.
            query_cost = """
            WITH RatedTransactions AS (
                SELECT 
                    ITEMNMBR,
                    UNITCOST,
                    DOCDATE,
                    ROW_NUMBER() OVER (PARTITION BY ITEMNMBR ORDER BY DOCDATE DESC, DEX_ROW_ID DESC) as rn
                FROM IV30300
                WHERE DOCDATE <= ?
                  AND DOCTYPE IN (4, 1) -- 4=Receipt, 1=Adjustment (Increase)
                  AND UNITCOST > 0
            )
            SELECT ITEMNMBR, UNITCOST as LastCost
            FROM RatedTransactions
            WHERE rn = 1
            """
            
            history_df = pd.read_sql(query_history, conn, params=[as_of_date])
            cost_df = pd.read_sql(query_cost, conn, params=[as_of_date])
            
            conn.close()
            
            return base_df, history_df, cost_df
            
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with st.spinner(f"Calculating inventory back to {target_date}..."):
        base_df, history_df, cost_df = fetch_inventory_data(target_date)

    if base_df.empty:
        st.warning("No base inventory data found for MAIN location.")
        return

    # Merge and Calculate
    df = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
    df = pd.merge(df, cost_df, on='ITEMNMBR', how='left')
    df['QtyChange'] = df['QtyChange'].fillna(0)
    
    # As Of Qty = Current (MAIN) - Change (MAIN)
    df['Quantity'] = df['CurrentQty'] - df['QtyChange']
    
    # Filter out zero quantities
    df = df[df['Quantity'] != 0]
    
    # Calculate Extended Cost
    # Use LastCost if available, else Current Cost
    df['Unit Cost'] = df['LastCost'].fillna(df['CURRCOST'])
    df['Extended Cost'] = df['Quantity'] * df['Unit Cost']
    
    # Formatting
    df_final = df[[
        'ITEMNMBR', 
        'ITEMDESC', 
        'Quantity', 
        'Unit Cost', 
        'Extended Cost', 
        'GLCode'
    ]].copy()
    
    df_final.columns = ['Item Number', 'Description', 'Quantity', 'Unit Cost', 'Extended Cost', 'GL Code']
    df_final = df_final.sort_values('Item Number')

    # Metrics
    total_val = df_final['Extended Cost'].sum()
    total_qty = df_final['Quantity'].sum()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Value (MAIN)", f"${total_val:,.2f}")
    m2.metric("Total Quantity", f"{total_qty:,.0f}")
    m3.metric("Items Count", len(df_final))

    # Preview
    st.dataframe(df_final, use_container_width=True)

    # Excel Download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='Perpetual Inventory', index=False)
        worksheet = writer.sheets['Perpetual Inventory']
        for idx, col in enumerate(df_final.columns):
            max_len = max(
                df_final[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            col_letter = chr(64 + idx + 1)
            worksheet.column_dimensions[col_letter].width = min(max_len, 50)
            
    buffer.seek(0)
    
    st.download_button(
        label=f"📥 Download Report ({target_date})",
        data=buffer,
        file_name=f"Inventory_Report_MAIN_{target_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

if __name__ == "__main__":
    render_page()
