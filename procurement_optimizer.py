
import json
import os
import pandas as pd
import datetime
import streamlit as st
import hashlib

# Import existing helpers for demand/market data
from market_insights import calculate_buying_signals, get_priority_raw_materials
from constants import RAW_MATERIAL_CLASS_CODES

VENDOR_QUOTES_FILE = "vendor_quotes.jsonl"
BROKER_QUOTES_FILE = "broker_quotes.jsonl"

def load_jsonl(filename):
    data = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
    return data

class ProcurementOptimizer:
    def __init__(self, cursor):
        self.cursor = cursor
        self.vendor_quotes = load_jsonl(VENDOR_QUOTES_FILE)
        self.broker_quotes = load_jsonl(BROKER_QUOTES_FILE)
        self.demand_data = None

    def _fetch_demand(self):
        """Get priority items (runway < 90 days or high usage)."""
        # We reuse the existing logic for priority materials
        df = get_priority_raw_materials(self.cursor, limit=5000, require_purchase_history=True) # Increased limit for broader analysis
        
        # Calculate runway for each
        items = []
        from market_insights import calculate_inventory_runway
        
        for _, row in df.iterrows():
            item_num = row['ITEMNMBR'].strip()
            runway_data = calculate_inventory_runway(self.cursor, item_num)
            
            # Fetch Buy Signal (Market Timing)
            buy_signal = calculate_buying_signals(self.cursor, item_num, runway_days=runway_data.get('runway_days'))
            signal_strength = buy_signal.get('signal', 'Hold')
            signal_score = buy_signal.get('score', 0)

            items.append({
                'Item': item_num,
                'Description': row.get('ITEMDESC', 'Unknown'),
                'CurrentCost': row.get('CURRCOST', 0), # Last PO / Current Avg
                'StandardCost': row.get('STNDCOST', 0),
                'RunwayDays': runway_data.get('runway_days', 999),
                'Urgency': runway_data.get('urgency', 'OK'),
                'Usage30d': runway_data.get('avg_daily_usage', 0) * 30, # Approx monthly usage
                'BuySignal': signal_strength,
                'SignalScore': signal_score
            })
        
        self.demand_data = pd.DataFrame(items)

    def optimize(self):
        """
        Match Demand -> Vendor Quotes -> Broker Rates to find Best Landed Cost.
        """
        if self.demand_data is None:
            self._fetch_demand()
            
        if self.demand_data.empty:
            return pd.DataFrame()

        # Index quotes for faster lookup
        # Vendor Quotes: List of quotes for each item
        vendor_map = {} # Item -> [Quotes]
        for q in self.vendor_quotes:
            item = q.get('item', '').strip()
            if item:
                if item not in vendor_map:
                    vendor_map[item] = []
                vendor_map[item].append(q)

        # Broker Quotes: Map by vendor_quote_id
        broker_map = {} # vendor_quote_id -> [Freight Bids]
        for b in self.broker_quotes:
            v_id = b.get('vendor_quote_id')
            if v_id:
                if v_id not in broker_map:
                    broker_map[v_id] = []
                # Only consider valid bids (date check could go here)
                broker_map[v_id].append(b)

        recommendations = []

        for _, demand in self.demand_data.iterrows():
            item = demand['Item']
            
            # 1. Find Matching Vendor Quotes
            potential_routes = vendor_map.get(item, [])
            
            best_option = None
            min_landed_cost = float('inf')

            for v_quote in potential_routes:
                # Material Price per Unit
                mat_price = float(v_quote.get('price', 0))
                if mat_price <= 0: continue
                
                # Check for Freight Bids
                freight_bids = broker_map.get(v_quote.get('id'), [])
                
                # Logic: Find best freight bid OR assume standard/market rate?
                # For now, if no freight bid, we can't truly calculate landed cost accurately 
                # unless we have a fallback. Let's start by analyzing confirmed routes only 
                # OR show "Price (Ex-Works)" if no freight.
                
                # Let's find lowest freight bid
                best_freight = 0
                freight_source = "Ex-Works (Pickup)"
                
                if freight_bids:
                    # Sort by price
                    sorted_freight = sorted(freight_bids, key=lambda x: float(x.get('freight_price', 0)))
                    best_bid = sorted_freight[0]
                    best_freight = float(best_bid.get('freight_price', 0))
                    freight_source = f"Freight via {best_bid.get('broker_id')}"
                else:
                    # HEURISTIC: Estimate freight if missing? 
                    # For now, define it as high/unquoted to penalty it, strictly preferring quoted routes
                    # BUT user wants to see "Optimized Route", so maybe showing the material price is still valuable.
                    # Let's flag it.
                    freight_source = "No Freight Quote"
                    best_freight = 0 # Can't add 0 cost, it's misleading.
                
                # Calculate Landed Cost Unit
                # Assumption: Truckload Quantity = 45,000 lbs (Standard Chem Tanker/Dry Van)
                # If unit is KG or GAL, this conversion needs to be smarter. 
                # For this specific user (Chemicals), lbs is common, or units.
                # Let's assume the 'price' is per UNIT.
                # Freight is per LOAD (usually).
                # Need load size. defaulting to 1 for calculation if unknown logic.
                # BETTER: Display Material Unit Price + Freight Total separately? 
                # User wants "Optimized Route and Price". 
                # Let's calculate Total Cost for a 'Standard Order' (e.g. 40,000 units/lbs) 
                
                STANDARD_ORDER_QTY = 40000 
                
                # Option A: Route with Freight
                if freight_bids:
                    total_mat_cost = mat_price * STANDARD_ORDER_QTY
                    total_landed = total_mat_cost + best_freight
                    unit_landed = total_landed / STANDARD_ORDER_QTY
                    
                    if unit_landed < min_landed_cost:
                        min_landed_cost = unit_landed
                        best_option = {
                            'Type': 'Verified Route',
                            'Vendor': v_quote.get('vendor'),
                            'Origin': v_quote.get('location'),
                            'Broker': freight_source,
                            'MaterialPrice': mat_price,
                            'FreightTotal': best_freight,
                            'LandedCost': unit_landed,
                            'LeadTime': v_quote.get('lead_time', '?'),
                            'Packaging': v_quote.get('packaging', 'Unknown'),
                            'Equipment': best_bid.get('equipment_type', 'Standard') if freight_bids else 'Required',
                            'Savings': demand['CurrentCost'] - unit_landed
                        }

                # Option B: Material Only (if it's cheap enough to warrant looking for freight)
                elif best_option is None:
                     # Keep as backup if no fully quoted route exsists
                     best_option = {
                            'Type': 'Material Only',
                            'Vendor': v_quote.get('vendor'),
                            'Origin': v_quote.get('location'),
                            'Broker': 'Needs Freight',
                            'MaterialPrice': mat_price,
                            'FreightTotal': 0,
                            'LandedCost': mat_price, # Misleading but base
                            'LeadTime': v_quote.get('lead_time', '?'),
                            'Savings': 0
                     }

            # Determine Recommended Action based on all factors
            # Factors: Landed Cost vs Market, Runway Urgency, Market Buy Signal
            
            recommendation_score = 0
            action_label = "Monitor"
            
            # Base Score from Buy Signal (Market Timing)
            recommendation_score += demand['SignalScore']
            
            if best_option:
                # Calculate True Savings (Standard Cost vs Landed Cost)
                # If Landed < Current, it's efficient.
                eff_savings = demand['CurrentCost'] - best_option['LandedCost']
                eff_pct = (eff_savings / demand['CurrentCost']) * 100 if demand['CurrentCost'] > 0 else 0
                
                best_option['EfficiencySavings'] = eff_savings
                best_option['EfficiencyPct'] = eff_pct
                
                # Boost score if price is efficient
                if eff_pct > 0:
                    recommendation_score += 20
                if eff_pct > 10:
                    recommendation_score += 20
                    
                # Action Logic
                if demand['RunwayDays'] < 30:
                    action_label = "CRITICAL BUY" # Must buy, pick best route
                elif recommendation_score > 80:
                    action_label = "STRONG BUY" # Good market + Good quote
                elif recommendation_score > 60:
                    action_label = "BUY"
                else:
                    action_label = "Weak Buy / Hold"
                    
                recommendations.append({
                    **demand,
                    **best_option,
                    'OptimizationScore': recommendation_score,
                    'ActionLabel': action_label
                })

        return pd.DataFrame(recommendations)

def render_procurement_cockpit(cursor):
    st.title(">> PROCUREMENT_COMMAND_CENTER")
    st.caption("Unified view of Demand, Supply, and Logistics optimization.")
    
    optimizer = ProcurementOptimizer(cursor)
    
    # 1. Critical Alerts (Demand Side)
    # Re-fetch strictly critical items from optimizer logic or just use what we have
    # We'll run the full optimization first as it fetches demand
    
    with st.spinner("Running global optimization engine..."):
        df_opt = optimizer.optimize()

    # SECTION 1: CRITICAL RUNWAY ALERTS
    # Highlighting items that NEED procurement now regardless of price
    st.markdown("### 1. CRITICAL_DEMAND_ALERTS")
    
    if optimizer.demand_data is not None and not optimizer.demand_data.empty:
        critical = optimizer.demand_data[optimizer.demand_data['RunwayDays'] < 30]
        if not critical.empty:
            for _, row in critical.iterrows():
                st.error(f"⚠️ **{row['Item']}** - {row['Description']} | Runway: {row['RunwayDays']:.1f} Days")
        else:
            st.success("Analysis complete: No critical inventory shortages detected (<30 days).")
    
    st.divider()

    # SECTION 2: OPTIMIZED ROUTES
    st.markdown("### 2. OPTIMIZED_SOURCING_MATRIX")
    st.caption("Showing Best Landed Cost options matching verified Vendor Quotes + Freight Bids.")

    if not df_opt.empty:
        # Filter for display
        # We want to show: Item | Needed By | Market Price | Best Option (Vendor+Broker) | Landed Cost | Savings
        
        # Sort by Savings (High) or Urgency
        df_opt = df_opt.sort_values(by='Savings', ascending=False)

        for _, rec in df_opt.iterrows():
            
            # Styling based on savings
            is_verified = rec['Type'] == 'Verified Route'
            savings_pct = (rec['Savings'] / rec['CurrentCost']) * 100 if rec['CurrentCost'] > 0 else 0
            
            card_color = "#859900" if savings_pct > 0 else "#b58900"
            border_style = "solid" if is_verified else "dashed"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{rec['Item']}** - {rec['Description']}")
                st.caption(f"Current Avg Cost: ${rec['CurrentCost']:.4f}")
            
            with col2:
                if is_verified:
                    st.markdown(f"**Best Route:** {rec['Vendor']} ➔ {rec['Broker']}")
                    st.markdown(f"Origin: {rec['Origin']} ({rec['LeadTime']} days)")
                    st.caption(f"Logistics: {rec.get('Packaging', '?')} via {rec.get('Equipment', '?')}")
                else:
                    st.markdown(f"**Vendor:** {rec['Vendor']}")
                    st.warning("Needs Freight Quote")
            
            with col3:
                st.metric(
                    "Landed Cost", 
                    f"${rec['LandedCost']:.4f}", 
                    delta=f"{rec['EfficiencyPct']:.1f}% vs Last PO"
                )
                
                # Dynamic Action Button
                btn_type = "primary" if "BUY" in rec['ActionLabel'] else "secondary"
                if st.button(f"{rec['ActionLabel']}: {rec['Item']}", key=f"btn_{rec['Item']}", type=btn_type):
                    preview_po_modal(rec)
            
            st.divider()
    else:
        st.info("No active vendor/broker routes found aligning with current demand.")
        
    # SECTION 3: RAW DATA ACCESS
    with st.expander("View Underlying Data"):
        st.subheader("Vendor Quotes")
        st.dataframe(pd.DataFrame(optimizer.vendor_quotes))
        st.subheader("Broker Bids")
        st.dataframe(pd.DataFrame(optimizer.broker_quotes))

@st.dialog("Purchase Order Preview")
def preview_po_modal(rec):
    st.markdown(f"### 📄 DRAFT PURCHASE ORDER")
    st.markdown("---")
    
    # Header
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**VENDOR:** {rec['Vendor']}")
        st.markdown(f"**SHIP FROM:** {rec['Origin']}")
    with c2:
        st.markdown(f"**DATE:** {datetime.date.today()}")
        st.markdown(f"**TERMS:** Net 30 (Standard)")
    
    st.markdown("---")
    
    # Line Items
    qty = rec.get('BuyQty', 45000) # Default to full truckload if missing
    unit_price = rec['MaterialCost']
    material_total = qty * unit_price
    
    st.markdown("#### PROPOSED LINE ITEMS")
    st.dataframe(pd.DataFrame([{
        "Item": rec['Item'],
        "Description": rec['Description'],
        "Quantity": qty,
        "UOM": "LBS", # Assumption for raw materials
        "Unit Price": f"${unit_price:.4f}",
        "Extended": f"${material_total:,.2f}"
    }]), hide_index=True)
    
    # Logistics
    st.markdown("#### LOGISTICS DETAIL")
    st.info(f"🚚 **F.O.B via {rec['Broker']}**: ${rec['FreightCost']:,.2f} Flat Rate")
    
    st.markdown("---")
    
    # Totals
    total_landed = material_total + rec['FreightCost']
    
    t1, t2 = st.columns([3, 1])
    with t2:
        st.markdown(f"**Subtotal:** ${material_total:,.2f}")
        st.markdown(f"**Freight:** ${rec['FreightCost']:,.2f}")
        st.markdown(f"### TOTAL: ${total_landed:,.2f}")
        
    st.markdown("---")
    
    b1, b2 = st.columns(2)
    with b1:
        if st.button("❌ Cancel", use_container_width=True):
            st.rerun()
    with b2:
        if st.button("✅ Release PO", type="primary", use_container_width=True):
            st.balloons()
            st.success(f"PO Released to {rec['Vendor']}!")
            # Logic to save PO would go here

