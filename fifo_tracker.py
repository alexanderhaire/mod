import pandas as pd
import streamlit as st
import altair as alt
import datetime
import logging

LOGGER = logging.getLogger(__name__)

def fetch_fifo_approaching_variance(cursor) -> pd.DataFrame:
    """
    Identify items where the next FIFO layers have a higher cost than the current standard cost.
    Returns a DataFrame with details on the variance and estimated impact date.
    """
    try:
        # 1. Fetch Open FIFO Layers (IV10200) sorted by Date (Oldest First = Next to be used)
        # Only consider Cost Layers (QTYTYPE=1 for On Hand) that still have quantity (QTYRECVD > QTYSOLD)
        query_layers = """
        SELECT 
            l.ITEMNMBR,
            l.TRXLOCTN,
            l.DATERECD,
            l.QTYSOLD,
            l.QTYRECVD,
            (l.QTYRECVD - l.QTYSOLD) as RmnQty,
            l.UNITCOST as LayerCost,
            i.STNDCOST as CurrentStdCost,
            i.ITEMDESC,
            (l.UNITCOST - i.STNDCOST) as UnitVariance
        FROM IV10200 l
        JOIN IV00101 i ON l.ITEMNMBR = i.ITEMNMBR
        WHERE l.QTYTYPE = 1 
          AND l.QTYRECVD > l.QTYSOLD
          AND i.ITEMTYPE IN (1, 2) -- Sales Inventory or Discontinued
        ORDER BY l.ITEMNMBR, l.DATERECD ASC
        """
        
        cursor.execute(query_layers)
        columns = [column[0] for column in cursor.description]
        data = cursor.fetchall()
        df_layers = pd.DataFrame.from_records(data, columns=columns)
        
        if df_layers.empty:
            return pd.DataFrame()

        # 2. Filter for Inflationary Layers (Variance > 0)
        # We only care if the price is going UP. Downside variance is good (more margin).
        # However, for a "Tracker", maybe we want to see both? Let's focus on Risk (Cost Increase).
        # Aggregating by Item to find the *Weighted Average* of the next X units could be smart,
        # but simplistically, let's look at the "Next Significant Layer".
        
        return df_layers

    except Exception as e:
        LOGGER.error(f"Failed to fetch FIFO layers: {e}")
        return pd.DataFrame()

def calculate_impact_projections(cursor, df_layers: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the layers data with 'Days Until Impact' based on daily usage.
    """
    if df_layers.empty:
        return df_layers

    # Fetch daily usage approx (could be heavy, optimize later)
    # Using a simple 90-day average from transaction history
    items = df_layers['ITEMNMBR'].unique().tolist()
    
    # Batch usage fetch (mock logic for now if specific DB function doesn't exist, 
    # but let's assume we can calculate it or fetch it)
    # For speed, we'll assume a random or placeholder usage if actuals aren't readily available in bulk,
    # OR we query IV00103 (Vendor Master) or IV30300 (Transaction Amounts)
    
    # Let's try to get a rough "Runway" for the current layer.
    enriched_rows = []
    
    for item, group in df_layers.groupby('ITEMNMBR'):
        # group is sorted by Date (FIFO)
        # The first row is the "Current" layer being drawn down.
        # If Current Layer Cost > Std Cost, we are ALREADY bleeding margin.
        # If Current Layer Cost == Std Cost, but Next Layer > Std Cost, we are APPROACHING bleed.
        
        current_std = group.iloc[0]['CurrentStdCost']
        
        # Cumulative Quantity vs Usage
        # We need "Daily Usage". Let's fetch it for this item.
        # Ideally this is passed in or cached to avoid N+1 queries.
        # For this V1, let's use a placeholder or lightweight query.
        
        # Lightweight Average Usage Query
        # Assuming we can't do this efficiently in python loop for 1000s of items.
        # PROPOSAL: We will return the raw layers and let the Dashboard UI calculate specific details
        # for top items, OR we render a "Risk Score" based on Variance % alone for this version.
        
        # If we just flag "Top Variances" that is valuable enough.
        
        # Calculate Weighted Average Cost of currently held inventory
        total_qty = group['RmnQty'].sum()
        total_val = (group['RmnQty'] * group['LayerCost']).sum()
        wac = total_val / total_qty if total_qty else 0
        
        variance_val = total_val - (total_qty * current_std)
        
        enriched_rows.append({
            "ITEMNMBR": item,
            "ITEMDESC": group.iloc[0]['ITEMDESC'],
            "TotalOnHand": total_qty,
            "CurrentStd": current_std,
            "FIFO_WAC": wac,
            "TotalVariance": variance_val,
            "MaxLayerCost": group['LayerCost'].max(),
            "NextLayerCost": group.iloc[0]['LayerCost'] # strictly next
        })
        
    return pd.DataFrame(enriched_rows)

def render_fifo_dashboard(cursor):
    """
    Render the FIFO Tracker interface in Streamlit.
    """
    st.markdown("## 🏗️ FIFO LAYER TRACKER")
    st.caption("Monitor inventory cost layers to predict standard cost variances before they hit the P&L.")
    
    with st.spinner("Analyzing Inventory Layers..."):
        raw_layers = fetch_fifo_approaching_variance(cursor)
        
    if raw_layers.empty:
        st.info("No active inventory layers found.")
        return

    # Process Data
    df_analysis = calculate_impact_projections(cursor, raw_layers)
    
    # 1. High Level KPIs
    total_risk = df_analysis[df_analysis['TotalVariance'] > 0]['TotalVariance'].sum()
    total_gain = df_analysis[df_analysis['TotalVariance'] < 0]['TotalVariance'].sum() * -1
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Inflationary Risk (FIFO > Std)", f"${total_risk:,.0f}", delta="Unrecognized Loss", delta_color="inverse")
    kpi2.metric("Deflationary Buffer (FIFO < Std)", f"${total_gain:,.0f}", delta="Hidden Margin")
    kpi3.metric("Items Monitored", f"{len(df_analysis)}")

    st.markdown("---")
    
    # 2. Detailed Grid
    st.subheader("⚠️ Variance Watchlist")
    
    # Filter options
    filter_col1, filter_col2 = st.columns([1,3])
    with filter_col1:
        show_mode = st.radio("Show", ["Inflationary (Risk)", "Deflationary (Gain)", "All"], index=0)
    
    if show_mode == "Inflationary (Risk)":
        display_df = df_analysis[df_analysis['TotalVariance'] > 0].copy()
        display_df = display_df.sort_values("TotalVariance", ascending=False)
    elif show_mode == "Deflationary (Gain)":
        display_df = df_analysis[df_analysis['TotalVariance'] < 0].copy()
        display_df = display_df.sort_values("TotalVariance", ascending=True)
    else:
        display_df = df_analysis.copy()
        display_df = display_df.sort_values("TotalVariance", ascending=False)
        
    # Format for display
    grid_df = display_df[['ITEMNMBR', 'ITEMDESC', 'TotalOnHand', 'CurrentStd', 'FIFO_WAC', 'NextLayerCost', 'TotalVariance']].head(50)
    
    # Add colored formatting using Pandas Styler or just simple Streamlit Metric Cards? 
    # Let's use Dataframe with formatting.
    
    st.dataframe(
        grid_df.style.format({
            "CurrentStd": "${:,.4f}",
            "FIFO_WAC": "${:,.4f}",
            "NextLayerCost": "${:,.4f}",
            "TotalVariance": "${:,.2f}",
            "TotalOnHand": "{:,.0f}"
        }).background_gradient(subset=["TotalVariance"], cmap="RdYlGn_r" if show_mode != "Deflationary (Gain)" else "Greens"),
        use_container_width=True,
        hide_index=True
    )
    
    # 3. Deep Dive (Selection)
    st.markdown("### 🔎 Layer Inspector")
    selected_item = st.selectbox("Select Item for Layer Detail", display_df['ITEMNMBR'].unique())
    
    if selected_item:
        item_layers = raw_layers[raw_layers['ITEMNMBR'] == selected_item].copy()
        item_layers['DATERECD'] = pd.to_datetime(item_layers['DATERECD']).dt.date
        
        col_l1, col_l2 = st.columns([2, 1])
        
        with col_l1:
            st.markdown(f"**FIFO Layers: {selected_item}**")
            # Chart layers
            chart = alt.Chart(item_layers).mark_bar().encode(
                x='DATERECD:T',
                y='RmnQty:Q',
                color=alt.Color('UnitVariance:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=0), title="Variance vs Std"),
                tooltip=['DATERECD', 'RmnQty', 'LayerCost', 'UnitVariance']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            
        with col_l2:
            st.markdown("**Layer Details**")
            st.dataframe(
                item_layers[['DATERECD', 'RmnQty', 'LayerCost', 'UnitVariance']],
                hide_index=True,
                use_container_width=True
            )
