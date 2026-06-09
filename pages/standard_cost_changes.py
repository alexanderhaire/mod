import streamlit as st
import pandas as pd
import datetime
from db_pool import get_cursor
from ui_utils import format_currency, render_kpi_card

def get_cost_changes(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    # GP does not natively retain history of manual STNDCOST edits for FIFO items.
    # However, we can logically derive the historical standard cost by looking at the 
    # OSTDCOST (Original Standard Cost) snapshot that GP saves on every PO Receipt Line (POP10500).
    query = """
    WITH StartReceipts AS (
        SELECT 
            ITEMNMBR,
            OSTDCOST as StartCost,
            ROW_NUMBER() OVER(PARTITION BY ITEMNMBR ORDER BY DATERECD DESC, DEX_ROW_ID DESC) as rn
        FROM POP10500
        WHERE DATERECD <= ? AND OSTDCOST > 0
    ),
    -- Try to infer when the cost changed by finding the first receipt with the new cost
    FirstNewCost AS (
        SELECT 
            ITEMNMBR,
            OSTDCOST,
            MIN(DATERECD) as InferredChangeDate
        FROM POP10500
        WHERE DATERECD >= ?
        GROUP BY ITEMNMBR, OSTDCOST
    )
    SELECT 
        i.ITEMNMBR,
        i.ITEMDESC,
        s.StartCost as PREVCOST,
        i.STNDCOST as PRESENTCOST,
        COALESCE(f.InferredChangeDate, i.MODIFDT) as CHANGEDATE_I,
        '1900-01-01 00:00:00' as TIME1,
        'Unknown' as CHANGEBY_I,
        i.ITMCLSCD,
        'Derived from PO Receipts' as Source
    FROM IV00101 i
    JOIN StartReceipts s ON i.ITEMNMBR = s.ITEMNMBR AND s.rn = 1
    LEFT JOIN FirstNewCost f ON i.ITEMNMBR = f.ITEMNMBR AND f.OSTDCOST = i.STNDCOST
    WHERE s.StartCost <> i.STNDCOST
      -- We filter by MODIFDT or InferredChangeDate to catch the change
      AND (i.MODIFDT >= ? OR f.InferredChangeDate >= ?)
      AND (i.MODIFDT <= ? OR f.InferredChangeDate <= ?)
      
    UNION ALL

    
    -- Keep the native Rollup Utility history for actual standard cost items
    SELECT 
        h.ITEMNMBR,
        i.ITEMDESC,
        h.PREVCOST,
        h.PRESENTCOST,
        h.CHANGEDATE_I,
        h.TIME1,
        h.CHANGEBY_I,
        i.ITMCLSCD,
        'Rollup Utility' as Source
    FROM IV00118 h
    LEFT JOIN IV00101 i ON h.ITEMNMBR = i.ITEMNMBR
    WHERE h.CHANGEDATE_I >= ? AND h.CHANGEDATE_I <= ?
    ORDER BY CHANGEDATE_I DESC
    """
    
    try:
        with get_cursor() as cursor:
            s_date = start_date.strftime('%Y-%m-%d')
            e_date = end_date.strftime('%Y-%m-%d')
            
            cursor.execute(
                query, 
                s_date, s_date,           # CTEs
                s_date, s_date,           # >= dates
                e_date, e_date,           # <= dates
                s_date, e_date            # UNION ALL dates
            )
            columns = [column[0] for column in cursor.description]
            data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            df = pd.DataFrame(data)
            
            if not df.empty:
                df['TIME_STR'] = df['TIME1'].apply(
                    lambda x: x.strftime('%H:%M:%S') if pd.notnull(x) and x.year > 1900 else ''
                )
                df['CHANGEDATE_STR'] = df['CHANGEDATE_I'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                )
                df['CHANGE_TIMESTAMP'] = df['CHANGEDATE_STR'] + ' ' + df['TIME_STR']
                
                df['VARIANCE'] = df.apply(
                    lambda row: row['PRESENTCOST'] - row['PREVCOST'] if pd.notnull(row['PREVCOST']) else None, 
                    axis=1
                )
            
            return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def render_standard_cost_changes_page():
    st.markdown('<h1 style="color: #ffb000;">>> STANDARD_COST_CHANGES</h1>', unsafe_allow_html=True)
    st.write("View historical standard cost changes. Note: Direct manual edits on the Item Card do not track the previous cost.")

    # Date filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        return

    with st.spinner("Fetching cost changes..."):
        df = get_cost_changes(start_date, end_date)

    if df.empty:
        st.info("No cost changes found for the selected date range.")
        return

    # Filter by Item Class (optional)
    classes = ['ALL'] + sorted(df['ITMCLSCD'].dropna().unique().tolist())
    selected_class = st.selectbox("Filter by Item Class", classes)

    if selected_class != 'ALL':
        df = df[df['ITMCLSCD'] == selected_class]

    # Metrics
    total_changes = len(df)
    items_changed = df['ITEMNMBR'].nunique()
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        render_kpi_card("Total Modfications", str(total_changes))
    with col_m2:
        render_kpi_card("Unique Items", str(items_changed))

    st.markdown("### 📋 COST CHANGE HISTORY")
    
    # Text Search
    search = st.text_input("Search Items", "").upper()
    if search:
        df = df[df['ITEMNMBR'].str.contains(search) | df['ITEMDESC'].str.contains(search)]
        
    display_df = df[['CHANGE_TIMESTAMP', 'ITEMNMBR', 'ITEMDESC', 'Source', 'PREVCOST', 'PRESENTCOST', 'VARIANCE', 'CHANGEBY_I', 'ITMCLSCD']].copy()
    
    st.dataframe(
        display_df,
        column_config={
            "CHANGE_TIMESTAMP": st.column_config.TextColumn("Timestamp"),
            "ITEMNMBR": st.column_config.TextColumn("Item"),
            "ITEMDESC": st.column_config.TextColumn("Description"),
            "Source": st.column_config.TextColumn("Change Source"),
            "PREVCOST": st.column_config.NumberColumn("Previous Cost", format="%.5f"),
            "PRESENTCOST": st.column_config.NumberColumn("New Cost", format="%.5f"),
            "VARIANCE": st.column_config.NumberColumn("Variance", format="%.5f"),
            "CHANGEBY_I": st.column_config.TextColumn("User"),
            "ITMCLSCD": st.column_config.TextColumn("Class"),
        },
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    render_standard_cost_changes_page()
