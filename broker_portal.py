
import streamlit as st
import pandas as pd
import json
import datetime
import os
import hashlib
import re

from market_insights import get_priority_raw_materials, calculate_inventory_runway

VENDOR_QUOTES_FILE = "vendor_quotes.jsonl"
BROKER_QUOTES_FILE = "broker_quotes.jsonl"

def fetch_active_vendor_quotes():
    """Read vendor quotes from JSONL file."""
    quotes = []
    if os.path.exists(VENDOR_QUOTES_FILE):
        try:
            with open(VENDOR_QUOTES_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            q = json.loads(line)
                            # Basic validation or filtering for "active" could go here
                            # For now, just load everything.
                            # Generate a simple ID for referencing
                            q_str = json.dumps(q, sort_keys=True)
                            q['id'] = hashlib.md5(q_str.encode('utf-8')).hexdigest()
                            quotes.append(q)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            st.error(f"Error reading vendor quotes: {e}")
    return quotes

def get_priority_item_set(cursor):
    """Identify items that are in high demand (Runway < 90 days)."""
    if not cursor: return {}
    try:
        # Get priority materials (active/used)
        df = get_priority_raw_materials(cursor, limit=5000, require_purchase_history=True)
        # Map item -> runway_days
        priority_map = {}
        
        for _, row in df.iterrows():
            item = row['ITEMNMBR'].strip()
            # Check runway
            runway = calculate_inventory_runway(cursor, item)
            days = runway.get('runway_days', 999)
            if days < 90:
                priority_map[item] = days
                
        return priority_map
    except Exception as e:
        st.warning(f"Could not fetch demand signals: {e}")
        return {}

def save_freight_quote(broker_id, vendor_quote_id, vendor_quote_summary, freight_price, valid_until, notes, equipment_type="Unknown"):
    """Save the broker's freight quote."""
    record = {
        "broker_id": broker_id,
        "vendor_quote_id": vendor_quote_id,
        "vendor_quote_summary": vendor_quote_summary, # Snapshot for context
        "freight_price": freight_price,
        "valid_until": valid_until.isoformat() if valid_until else None,
        "notes": notes,
        "equipment_type": equipment_type,
        "submitted_at": datetime.datetime.now().isoformat()
    }
    
    with open(BROKER_QUOTES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def fetch_brokers(cursor):
    """Fetch active brokers (vendors) from PM00200."""
    if not cursor:
        return []
    try:
        # Reusing the vendor table as brokers are likely set up as vendors too
        cursor.execute("SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDSTTS = 1 ORDER BY VENDNAME")
        return cursor.fetchall()
    except Exception as e:
        st.error(f"Error loading broker list: {e}")
        return []

def fetch_estimated_origin(cursor, item_nmbr):
    """
    Estimate origin zip by looking up:
    1. Primary Vendor (IV00103)
    2. Last Vendor (POP30300)
    """
    if not cursor: return "TBD"
    try:
        # 1. Primary Vendor
        query_primary = """
        SELECT TOP 1 v.ZIPCODE 
        FROM IV00103 iv
        JOIN PM00200 v ON iv.VENDORID = v.VENDORID
        WHERE iv.ITEMNMBR = ? AND iv.ITMVNDTY = 1
        """
        cursor.execute(query_primary, item_nmbr)
        row = cursor.fetchone()
        if row and row[0]:
            return row[0].strip()[:5]

        # 2. Last Vendor from Purchase History
        query_history = """
        SELECT TOP 1 v.ZIPCODE
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        JOIN PM00200 v ON h.VENDORID = v.VENDORID
        WHERE l.ITEMNMBR = ?
        ORDER BY h.RECEIPTDATE DESC
        """
        cursor.execute(query_history, item_nmbr)
        row = cursor.fetchone()
        if row and row[0]:
            return row[0].strip()[:5]
            
    except Exception as e:
        pass # Fallback
        
    return "TBD"

def render_broker_portal(cursor=None):
    st.title(f"Freight Broker Portal")
    
    # 1. Identity Selection
    # 1. Verified Identity (from Login)
    selected_broker_id = st.session_state.user
    # Fallback if user_name wasn't set (legacy login session)
    selected_broker_name = st.session_state.get("user_name", selected_broker_id)
    
    st.markdown(f"Submitting quotes as: **{selected_broker_name}**")
    st.markdown("View active material routes and submit your freight pricing.")

    # 2. Fetch and Display Vendor Quotes (Routes)
    quotes = fetch_active_vendor_quotes()
    
    if not quotes:
        st.info("No active vendor routes available at this time.")
        return

    # Convert to DataFrame for easier sorting/display
    df_quotes = pd.DataFrame(quotes) if quotes else pd.DataFrame(columns=['item', 'price', 'vendor', 'location', 'distance_miles', 'id'])
    
    # Retroactive Filter: Remove any REC or PMX items from legacy quotes
    if not df_quotes.empty:
        df_quotes = df_quotes[~df_quotes['item'].str.startswith("REC-")]
        df_quotes = df_quotes[~df_quotes['item'].str.startswith("PMX")]
    
    # --- HYBRID PRICING LOGIC ---
    # User Request: Use current internal pricing (CURRCOST) initially, then transition to vendor prices as they come in.
    # We inject "Synthetic Quotes" for priority items that don't have an active vendor quote yet.
    
    if cursor:
        try:
            # Get all priority items (demand)
            df_priority = get_priority_raw_materials(cursor, limit=50, require_purchase_history=True)
            
            new_rows = []
            existing_quoted_items = set(df_quotes['item'].unique()) if not df_quotes.empty else set()
            
            for _, row in df_priority.iterrows():
                item_num = row['ITEMNMBR'].strip()
                
                # If this priority item has NO active vendor quote, inject internal cost
                if item_num not in existing_quoted_items:
                    # Create synthetic quote based on internal pricing
                    internal_price = float(row['CURRCOST'])
                    
                    # Estimate Origin
                    est_zip = fetch_estimated_origin(cursor, item_num)
                    
                    # Generate stable ID for bidding
                    start_str = f"internal_{item_num}_{internal_price}"
                    syn_id = hashlib.md5(start_str.encode('utf-8')).hexdigest()
                    
                    new_rows.append({
                        "item": item_num,
                        "price": internal_price,
                        "vendor": "Internal Estimate (Target)", # Label to distinguish
                        "location": f"Supplier Network ({est_zip})", # Show estimated Zip
                        "distance_miles": 0,
                        "id": syn_id,
                        "is_internal": True, # Flag for UI styling
                        "submitted_at": datetime.datetime.now().isoformat()
                    })
            
            if new_rows:
                df_internal = pd.DataFrame(new_rows)
                df = pd.concat([df_quotes, df_internal], ignore_index=True)
            else:
                df = df_quotes
                
        except Exception as e:
            st.warning(f"Error merging internal pricing: {e}")
            df = df_quotes
    else:
        df = df_quotes

    
    # Ensure columns exist (handle legacy data missing new fields)
    if 'distance_miles' not in df.columns:
        df['distance_miles'] = None
    if 'location' not in df.columns:
        df['location'] = "Unknown"
    if 'item' not in df.columns:
        df['item'] = "Unknown"
    if 'price' not in df.columns:
        df['price'] = 0.0
    if 'submitted_at' not in df.columns:
        # Fallback for old records
        df['submitted_at'] = datetime.datetime.min.isoformat()

    # Sort by Date (Newest First), then Price
    df = df.sort_values(by=['submitted_at', 'price'], ascending=[False, True])
    
    # 3. Smart Filtering (Priority Matches)
    filter_mode = st.radio("Show Opportunities", ["All Routes", "Priority Only (Demand + Best Price)"], index=1, horizontal=True)
    
    # Store runway info for sorting
    df['runway_days'] = 9999.0 
    
    if cursor:
        with st.spinner("Analyzing demand signals (Buy Calendar)..."):
            priority_map = get_priority_item_set(cursor) # Returns {item: runway_days}
            
            # 3a. Map runway to dataframe
            # This allows us to sort EVERYTHING by urgency
            df['runway_days'] = df['item'].map(priority_map).fillna(9999.0)
            
            if filter_mode == "Priority Only (Demand + Best Price)":
                # Logic: 
                # 1. Item must be in priority_map
                # 2. Quote must be the CHEAPEST for that item (Optimized Route)
                
                # Identify best price per item
                best_prices = {}
                for _, row in df.iterrows():
                    item = row['item']
                    price = float(row['price'])
                    if item not in best_prices or price < best_prices[item]:
                        best_prices[item] = price
                
                # Filter
                filtered_indices = []
                for idx, row in df.iterrows():
                    item = row['item']
                    price = float(row['price'])
                    
                    is_needed = item in priority_map
                    is_best_price = price <= best_prices.get(item, float('inf'))
                    
                    if is_needed and is_best_price:
                        filtered_indices.append(idx)
                
                if filtered_indices:
                    df = df.loc[filtered_indices]
                    st.success(f"Found {len(df)} golden opportunities matching demand.")
                else:
                    st.info("No priority routes found matching current demand criteria.")
                    df = pd.DataFrame() # Empty

    # Sort Strategy:
    # 1. URGENCY (Runway Days Ascending) - "Buy Calendar" order
    # 2. PRICE (Lowest First)
    if 'runway_days' in df.columns:
        df = df.sort_values(by=['runway_days', 'price'], ascending=[True, True])
    else:
        # Fallback
        df = df.sort_values(by="price", ascending=True)

    
    # Selection UI
    st.subheader("Available Routes")
    
    # Helper to clean up zip display: "Origin Zip -> Dest Zip"
    def _extract_zip(loc_str):
        if not loc_str: return "Unknown"
        # If explicitly TBD or Internal
        if "TBD" in loc_str or "Supplier Network" in loc_str:
            return "TBD"
        # Try to find a 5 digit zip in the string
        match = re.search(r'\b\d{5}\b', loc_str)
        if match:
            return match.group(0)
        # Fallback to first part of address
        return loc_str.split(',')[0].strip()

    def format_func(row):
        origin_zip = _extract_zip(row.get('location'))
        dest_zip = "33563" # Default Plant City
        dist_str = f"{row.get('distance_miles')} mi" if row.get('distance_miles') else "? mi"
        runway_str = f" | ⏳ {row.get('runway_days'):.0f}d Runway" if row.get('runway_days', 9999) < 999 else ""
        
        return f"{origin_zip} ➔ {dest_zip} | {row.get('item')} | ${row.get('price', 0):.2f} {runway_str}"
    
    # Create list of options corresponding to rows
    route_options = df.to_dict('records')
    selected_route = st.selectbox(
        "Select a Route to Quote Freight", 
        options=route_options,
        format_func=format_func,
        help="Select the material shipment you want to bid on."
    )

    if selected_route:
        st.divider()
        st.subheader("Submit Freight Quote")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Target Shipment:**")
            
            # ZIP VISUALIZATION
            origin_zip = _extract_zip(selected_route.get('location'))
            st.markdown(f"### {origin_zip} ➔ 33563")
            st.caption(f"Origin: {selected_route.get('location')} | Dest: Plant City, FL")
            
            st.divider()
            
            st.write(f"**Item:** {selected_route.get('item')}")
            st.write(f"**Material Cost:** ${selected_route.get('price'):.2f}")
            st.write(f"**Vendor:** {selected_route.get('vendor')}")
            
            if selected_route.get('is_internal'):
                st.info("ℹ️ **Target Route:** Internal estimate. Loc/Vendor TBD.")
            
            # DOCUMENT LINKS (SDS / COA)
            st.markdown("---")
            st.markdown("**Vendor Documents:**")
            
            sds_path = selected_route.get('sds_path')
            coa_path = selected_route.get('coa_path')
            
            has_docs = False
            if sds_path and os.path.exists(sds_path):
                has_docs = True
                with open(sds_path, "rb") as f:
                    st.download_button("📄 Download SDS", f, file_name=os.path.basename(sds_path))
            
            if coa_path and os.path.exists(coa_path):
                has_docs = True
                with open(coa_path, "rb") as f:
                    st.download_button("🧪 Download COA", f, file_name=os.path.basename(coa_path))
            
            if not has_docs:
                st.caption("No documents uploaded by vendor yet.")
            
        with col2:
            freight_price = st.number_input("Freight Cost ($ Total)", min_value=0.0, step=10.0, format="%.2f")
            valid_until = st.date_input("Quote Valid Until", min_value=datetime.date.today(), key="freight_valid")
            
            # Equipment Configuration
            equipment_opts = ["Liquid Tanker (Stainless)", "Liquid Tanker (Rubber Lined)", "Dry Van", "Hopper/Dump", "Reefer", "Flatbed", "Other"]
            equipment_type = st.selectbox("Equipment Type", equipment_opts)
            
            notes = st.text_area("Notes (Equipment type, availability, etc.)")
            
            if st.button("Submit Freight Bid", type="primary"):
                if freight_price > 0:
                    summary = {
                        "item": selected_route.get('item'),
                        "origin": selected_route.get('location'),
                        "distance": selected_route.get('distance_miles'),
                        "material_price": selected_route.get('price')
                    }
                    save_freight_quote(
                        selected_broker_id, 
                        selected_route.get('id'), 
                        summary, 
                        freight_price, 
                        valid_until, 
                        notes,
                        equipment_type
                    )
                    st.success(f"Freight bid submitted successfully for {selected_broker_name}!")
                else:
                    st.error("Please enter a valid freight cost.")

    # History Section
    st.divider()
    st.subheader("Your Submitted Bids")
    if os.path.exists(BROKER_QUOTES_FILE):
        try:
            my_bids = []
            with open(BROKER_QUOTES_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        b = json.loads(line)
                        if b.get('broker_id') == st.session_state.user:
                            my_bids.append(b)
            
            if my_bids:
                bid_df = pd.DataFrame(my_bids)
                # Flatten summary for display if needed, or just show main cols
                st.dataframe(bid_df[['submitted_at', 'freight_price', 'notes', 'valid_until']])
            else:
                st.info("You haven't submitted any bids yet.")
        except Exception:
            st.warning("Could not load bid history.")
