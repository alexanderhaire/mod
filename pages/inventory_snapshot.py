import streamlit as st
import pandas as pd
from datetime import date, timedelta
from db_pool import get_connection
from constants import PRIMARY_LOCATION

st.set_page_config(layout="wide", page_title="Inventory Snapshot")

def get_data(items, location, date_start, date_end, search_all_locations):
    if not items:
        return pd.DataFrame(), []
        
    messages = []
    
    try:
        with get_connection() as conn:
            # 0. Validate Items in IV00101
            placeholders = ",".join("?" for _ in items)
            val_query = f"SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITEMNMBR IN ({placeholders})"
            df_val = pd.read_sql(val_query, conn, params=items)
            
            found_items = set(df_val['ITEMNMBR'].str.strip())
            missing_items = set(items) - found_items
            
            if missing_items:
                messages.append(f"⚠️ **Warning**: The following items were not found in the Item Master (IV00101): {', '.join(missing_items)}")
            
            # 1. Current On Hand (IV00102)
            if search_all_locations:
                loc_clause = ""
                loc_params = items
                select_clause = "SUM(QTYONHND) as QTYONHND"
                group_clause = "GROUP BY ITEMNMBR"
                messages.append("ℹ️ Aggregating quantities from **ALL** locations.")
            else:
                loc_clause = "AND LOCNCODE = ?"
                loc_params = [*items, location]
                select_clause = "QTYONHND"
                group_clause = ""
            
            oh_query = f"""
            SELECT ITEMNMBR, {select_clause}
            FROM IV00102
            WHERE ITEMNMBR IN ({placeholders}) {loc_clause}
            {group_clause}
            """
            
            df_current = pd.read_sql(oh_query, conn, params=loc_params)
            
            # Map descriptions
            desc_map = dict(zip(df_val['ITEMNMBR'].str.strip(), df_val['ITEMDESC'].str.strip()))
            
            # 2. Adjustments for HISTORICAL Calculations
            # HistOH(Date) = CurrentOH - Sum(NetChange where DOCDATE > Date)
            
            if search_all_locations:
                trx_loc_clause = ""
                trx_base_params = items
            else:
                trx_loc_clause = "AND TRXLOCTN = ?"
                trx_base_params = [*items, location]
                
            def get_adjustments_after(target_date):
                params = items + ([target_date] if search_all_locations else [target_date, location])
                q = f"""
                SELECT ITEMNMBR, SUM(TRXQTY) as NetChange
                FROM IV30300
                WHERE ITEMNMBR IN ({placeholders})
                  AND DOCDATE > ? 
                  {trx_loc_clause}
                GROUP BY ITEMNMBR
                """
                df = pd.read_sql(q, conn, params=params)
                df.set_index('ITEMNMBR', inplace=True)
                return df

            df_adj_start = get_adjustments_after(date_start)
            df_adj_end = get_adjustments_after(date_end)
            
            # 3. Activity between Dates (for display)
            # Receipts and Usage between start+1 and end
            usage_start = date_start + timedelta(days=1)
            usage_end = date_end
            
            if search_all_locations:
                period_params = [*items, usage_start, usage_end]
            else:
                period_params = [*items, usage_start, usage_end, location]

            # Query for Receipts (TRXQTY > 0)
            receipts_query = f"""
            SELECT ITEMNMBR, SUM(TRXQTY) as QtyRecieved
            FROM IV30300
            WHERE ITEMNMBR IN ({placeholders})
              AND DOCDATE >= ?
              AND DOCDATE <= ?
              AND TRXQTY > 0
              {trx_loc_clause}
            GROUP BY ITEMNMBR
            """
            
            # Query for Usage (TRXQTY < 0)
            usage_query = f"""
            SELECT ITEMNMBR, SUM(TRXQTY) as QtyUsed
            FROM IV30300
            WHERE ITEMNMBR IN ({placeholders})
              AND DOCDATE >= ?
              AND DOCDATE <= ?
              AND TRXQTY < 0
              {trx_loc_clause}
            GROUP BY ITEMNMBR
            """
            
            df_receipts = pd.read_sql(receipts_query, conn, params=period_params)
            df_receipts.set_index('ITEMNMBR', inplace=True)
            
            df_usage = pd.read_sql(usage_query, conn, params=period_params)
            df_usage.set_index('ITEMNMBR', inplace=True)
            
            # Combine Results
            results = []
            
            for itm in items:
                # Current System On Hand
                curr_oh = 0
                if not df_current.empty:
                    match = df_current[df_current['ITEMNMBR'].str.strip() == itm]
                    if not match.empty:
                        curr_oh = match['QTYONHND'].iloc[0]
                
                desc = desc_map.get(itm, "Unknown/Not Found")
                
                # Helper to get value safely
                def get_val(df_in, col):
                    if itm in df_in.index:
                        return df_in.loc[itm, col]
                    elif itm in df_in.index.str.strip():
                        return df_in.loc[df_in.index.str.strip() == itm, col].iloc[0]
                    return 0

                # Calc Snapshots
                adj_start = get_val(df_adj_start, 'NetChange')
                adj_end = get_val(df_adj_end, 'NetChange')
                
                hist_oh_start = curr_oh - adj_start
                hist_oh_end = curr_oh - adj_end
                
                # Activity
                qty_rec = get_val(df_receipts, 'QtyRecieved')
                qty_used = get_val(df_usage, 'QtyUsed')
                qty_used_abs = abs(qty_used)
                
                results.append({
                    'Item Number': itm,
                    'Description': desc,
                    f'GP End of {date_start}': hist_oh_start,
                    f'Received': qty_rec,
                    f'Used': qty_used_abs,
                    f'GP End of {date_end}': hist_oh_end,
                    'Today OH': curr_oh
                })
                
            return pd.DataFrame(results), messages

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), [f"❌ Critical Error: {str(e)}"]

# UI Layout
st.title("📦 Inventory Snapshot & Usage Report")

st.markdown("""
Generate a report showing:
1.  **On Hand Quantity** as of a specific date (back-calculated from today).
2.  **Usage** (Consumption) over a specific period.
""")

# Sidebar Controls
with st.sidebar:
    st.header("Configuration")
    
    # Locked to MAIN per user request
    st.info("🔒 Location locked to **MAIN**")
    location = PRIMARY_LOCATION
    search_all = False
    
    st.subheader("Comparison Dates")
    date_start = st.date_input("Start Date (End of Day)", value=date(2026, 1, 31))
    date_end = st.date_input("End Date (End of Day)", value=date(2026, 2, 3))
    
    st.info(f"Report will show Inventory as of **{date_start}** vs **{date_end}**, and activity between them.")

# Main Input
default_items = "CHEACETIC\nCHECITRIC\nCHEGLUCO\nCHEH2SO4\nCHELIGLIQ\nCHEMOLASS\nCHEMONETH\nCHESORB\nCL2CA34\nCL2CALIQ"
raw_items = st.text_area("Item Numbers (one per line)", height=300, value=default_items, placeholder="Paste items here...")

if st.button("Generate Report", type="primary"):
    # Strip whitespace from lines
    items = [line.strip() for line in raw_items.split('\n') if line.strip()]
    
    if not items:
        st.warning("Please enter at least one item number.")
    else:
        with st.spinner(f"Analyzing {len(items)} items..."):
            
            df, msgs = get_data(items, location, date_start, date_end, search_all)
            
            # Show messages
            for msg in msgs:
                st.info(msg)
                
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"inventory_snapshot_{date_start}_{date_end}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No data returned.")
