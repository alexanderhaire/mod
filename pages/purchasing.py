import streamlit as st
import pandas as pd
import datetime
import json
import os
from pathlib import Path
from db_pool import get_cursor
from constants import RAW_MATERIAL_CLASS_CODES
from ui_utils import format_currency, render_kpi_card

# Local storage path
OVERRIDE_FILE = Path(__file__).parent.parent / "data" / "purchasing_overrides.json"

def load_local_overrides():
    """Load overrides from local JSON file."""
    if not OVERRIDE_FILE.exists():
        OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
        return {}
    try:
        with open(OVERRIDE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_local_overrides(overrides):
    """Save overrides to local JSON file."""
    try:
        OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OVERRIDE_FILE, 'w') as f:
            json.dump(overrides, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Save Failed: {e}")
        return False

def get_purchasing_data():
    raw_material_classes = "', '".join(RAW_MATERIAL_CLASS_CODES)
    overrides = load_local_overrides()
    
    query = f"""
    WITH LatestOpenPO AS (
        SELECT 
            ITEMNMBR, 
            MAX(DEX_ROW_ID) as MaxID 
        FROM POP10110 
        GROUP BY ITEMNMBR
    ),
    OpenPOCost AS (
        SELECT 
            l.ITEMNMBR, 
            l.UNITCOST as OpenCost,
            h.VENDNAME as OpenVendor,
            h.DOCDATE as OpenDate
        FROM POP10110 l
        JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
        JOIN LatestOpenPO lo ON l.ITEMNMBR = lo.ITEMNMBR AND l.DEX_ROW_ID = lo.MaxID
    ),
    LatestHistPO AS (
        SELECT 
            ITEMNMBR, 
            MAX(DEX_ROW_ID) as MaxID 
        FROM POP30110 
        GROUP BY ITEMNMBR
    ),
    HistPOCost AS (
        SELECT 
            l.ITEMNMBR, 
            l.UNITCOST as HistCost,
            h.VENDNAME as HistVendor,
            h.DOCDATE as HistDate
        FROM POP30110 l
        JOIN POP30100 h ON l.PONUMBER = h.PONUMBER
        JOIN LatestHistPO lh ON l.ITEMNMBR = lh.ITEMNMBR AND l.DEX_ROW_ID = lh.MaxID
    )
    SELECT 
        i.ITEMNMBR, 
        i.ITEMDESC, 
        i.CURRCOST as GPCurrCost,
        o.OpenCost,
        o.OpenVendor,
        o.OpenDate,
        h.HistCost,
        h.HistVendor,
        h.HistDate,
        COALESCE(o.OpenCost, h.HistCost, i.CURRCOST) as TrueCost
    FROM IV00101 i
    LEFT JOIN OpenPOCost o ON i.ITEMNMBR = o.ITEMNMBR
    LEFT JOIN HistPOCost h ON i.ITEMNMBR = h.ITEMNMBR
    WHERE i.ITMCLSCD IN ('{raw_material_classes}')
    ORDER BY i.ITEMNMBR
    """
    
    try:
        with get_cursor() as cursor:
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            data = [dict(zip(columns, row)) for row in cursor.fetchall()]
            df = pd.DataFrame(data)
            
            # Inject locked costs from local storage
            df['LockedCost'] = df['ITEMNMBR'].apply(lambda x: overrides.get(x, {}).get('cost'))
            # Recalculate TrueCost with Locks as Priority
            df['TrueCost'] = df.apply(lambda r: r['LockedCost'] if pd.notnull(r['LockedCost']) else r['TrueCost'], axis=1)
            
            return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def save_overrides(df_changes):
    """Save manually locked costs back to local storage."""
    overrides = load_local_overrides()
    current_user = st.session_state.get("user", "default")
    now = datetime.datetime.now().isoformat()
    
    for _, row in df_changes.iterrows():
        item = row['ITEMNMBR']
        cost = row['LockedCost']
        
        if pd.notnull(cost):
            overrides[item] = {
                "cost": float(cost),
                "modified_by": current_user,
                "modified_date": now
            }
        else:
            if item in overrides:
                del overrides[item]
                
    if save_local_overrides(overrides):
        st.success("✅ Locked costs persisted to local storage.")

def render_purchasing_page():
    st.markdown('<h1 style="color: #ffb000;">>> PURCHASING_INTELLIGENCE</h1>', unsafe_allow_html=True)
    st.write("Extracting 'True Cost' for Raw Materials. Locked values prioritize Procurement Work/History.")

    # Initialize data
    if 'purchasing_df' not in st.session_state:
        with st.status("Fetching cost data from GP...", expanded=False):
            st.session_state.purchasing_df = get_purchasing_data()

    df = st.session_state.purchasing_df

    if df.empty:
        st.warning("No raw material data found or database error.")
        return

    # Metrics
    avg_true_cost = df['TrueCost'].mean()
    total_items = len(df)
    
    # Variance check
    df['Variance'] = df['TrueCost'] - df['GPCurrCost']
    top_movers = df[df['Variance'].abs() > 0.001].sort_values('Variance', ascending=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card("Raw Materials", str(total_items))
    with col2:
        render_kpi_card("Avg True Cost", format_currency(avg_true_cost, 4))
    with col3:
        render_kpi_card("Active Variances", str(len(top_movers)))

    st.markdown("### 📊 PROCUREMENT COMMAND CENTER")
    st.caption("Edit the 'LockedCost' column to override system values. Click 'Lock & Save' below.")
    
    # Search filter
    search = st.text_input("Search Items", "").upper()
    filtered_df = df.copy()
    if search:
        filtered_df = filtered_df[filtered_df['ITEMNMBR'].str.contains(search) | filtered_df['ITEMDESC'].str.contains(search)]

    # Data Editor
    edited_df = st.data_editor(
        filtered_df,
        column_order=["ITEMNMBR", "ITEMDESC", "LockedCost", "TrueCost", "GPCurrCost", "Variance", "OpenCost", "OpenVendor", "OpenDate", "HistCost", "HistVendor", "HistDate"],
        column_config={
            "ITEMNMBR": st.column_config.TextColumn("Item", disabled=True),
            "ITEMDESC": st.column_config.TextColumn("Description", disabled=True),
            "LockedCost": st.column_config.NumberColumn("🔒 LockedCost", format="%.5f", help="Manually override the cost for this item."),
            "TrueCost": st.column_config.NumberColumn("Final TrueCost", format="%.5f", disabled=True),
            "GPCurrCost": st.column_config.NumberColumn("GP Curr Cost", format="%.5f", disabled=True),
            "Variance": st.column_config.NumberColumn("Variance", format="%.5f", disabled=True),
            "OpenCost": st.column_config.NumberColumn("Open PO Cost", format="%.5f", disabled=True),
            "HistCost": st.column_config.NumberColumn("Hist PO Cost", format="%.5f", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        key="purchasing_editor"
    )

    # Save logic
    if st.button("💾 Lock & Save Changes", type="primary"):
        # Identify what actually changed
        if st.session_state.purchasing_editor.get("edited_rows"):
            changes = []
            for idx, vals in st.session_state.purchasing_editor["edited_rows"].items():
                item_number = filtered_df.iloc[idx]['ITEMNMBR']
                locked_val = vals.get('LockedCost')
                changes.append({'ITEMNMBR': item_number, 'LockedCost': locked_val})
            
            save_overrides(pd.DataFrame(changes))
            # Refresh data
            st.session_state.pop('purchasing_df')
            st.rerun()

    if not top_movers.empty:
        st.markdown("### ⚠️ HIGHEST COST VARIANCES")
        st.dataframe(
            top_movers[['ITEMNMBR', 'ITEMDESC', 'TrueCost', 'GPCurrCost', 'Variance']].head(10),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    render_purchasing_page()
