import streamlit as st
import pandas as pd
import datetime
import json
import os
from market_insights import classify_item_segment

VENDOR_QUOTES_FILE = "vendor_quotes.jsonl"
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

def save_quote(vendor, item, price, valid_until, notes, location="Unknown", distance=None, lead_time=None, packaging=None, sds_file=None, coa_file=None):
    """Append a new quote to the JSONL file."""
    
    # Handle File Uploads
    sds_path = None
    coa_path = None
    
    if sds_file:
        sds_filename = f"sds_{item}_{vendor}_{sds_file.name}".replace(" ", "_").replace("/", "-")
        sds_path = os.path.join(UPLOADS_DIR, sds_filename)
        with open(sds_path, "wb") as f:
            f.write(sds_file.getbuffer())
            
    if coa_file:
        coa_filename = f"coa_{item}_{vendor}_{coa_file.name}".replace(" ", "_").replace("/", "-")
        coa_path = os.path.join(UPLOADS_DIR, coa_filename)
        with open(coa_path, "wb") as f:
            f.write(coa_file.getbuffer())

    record = {
        "vendor": vendor,
        "item": item,
        "price": price,
        "valid_until": valid_until.isoformat(),
        "notes": notes,
        "submitted_at": datetime.datetime.now().isoformat(),
        "location": location,
        "distance_miles": distance,
        "lead_time": lead_time,
        "packaging": packaging,
        "sds_path": sds_path,
        "coa_path": coa_path
    }
    with open(VENDOR_QUOTES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def fetch_vendors(cursor):
    """Fetch active vendors from PM00200."""
    if not cursor:
        return []
    try:
        # VENDSTTS 1 = Active
        # Fetch Address details: ADDRESS1, ADDRESS2, CITY, STATE, ZIPCODE
        cursor.execute("""
            SELECT VENDORID, VENDNAME, ADDRESS1, ADDRESS2, CITY, STATE, ZIPCODE 
            FROM PM00200 
            WHERE VENDSTTS = 1 
            ORDER BY VENDNAME
        """)
        return cursor.fetchall()
    except Exception as e:
        st.error(f"Error loading vendors: {e}")
        return []

def fetch_raw_materials(cursor):
    """Fetch items from IV00101 and filter for Raw Materials."""
    if not cursor:
        return []
    try:
        # Fetch Sales Inventory items (ITEMTYPE = 1) along with Class Code
        cursor.execute("SELECT ITEMNMBR, ITEMDESC, ITMCLSCD FROM IV00101 WHERE ITEMTYPE = 1 ORDER BY ITEMNMBR")
        rows = cursor.fetchall()
        
        raw_materials = []
        for row in rows:
            item_nmbr = str(row[0]).strip()
            item_desc = str(row[1]).strip()
            itm_class = str(row[2]).strip()
            
            # Use shared market analysis logic to classify
            if classify_item_segment(itm_class) == "Raw Material":
                # User Exclusion: No REC or PMX items
                if not item_nmbr.startswith("REC-") and not item_nmbr.startswith("PMX"):
                    raw_materials.append(f"{item_nmbr} - {item_desc}")
        
        return raw_materials
    except Exception as e:
        st.error(f"Error loading items: {e}")
        return []

def render_vendor_portal(cursor=None):
    st.title(f"Vendor Portal: {st.session_state.user}")
    
    # Vendor Selection
    selected_vendor_id = None
    selected_vendor_name = None
    selected_vendor_address = ""
    
    # 1. Verified Identity (from Login)
    selected_vendor_id = st.session_state.user
    selected_vendor_name = st.session_state.get("user_name", selected_vendor_id)
    selected_vendor_address = ""
    
    if cursor:
        try:
            # Fetch address for this specific vendor only
            cursor.execute("""
                SELECT ADDRESS1, ADDRESS2, CITY, STATE, ZIPCODE 
                FROM PM00200 
                WHERE VENDORID = ?
            """, selected_vendor_id)
            row = cursor.fetchone()
            if row:
                parts = [
                    (row[0] or "").strip(),
                    (row[1] or "").strip(),
                    (row[2] or "").strip(),
                    (row[3] or "").strip(),
                    (row[4] or "").strip()
                ]
                selected_vendor_address = ", ".join([p for p in parts if p])
        except Exception as e:
            st.warning(f"Could not load vendor address: {e}")
    
    st.markdown(f"**Welcome, {selected_vendor_name} ({selected_vendor_id})**")

    st.markdown("Submit your latest pricing and availability updates below.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("New Quote")
        
        # If we have a selected vendor, use it. Otherwise fall back to user
        vendor_identity = selected_vendor_name or st.session_state.user
        st.info(f"Submitting as: **{vendor_identity}**")
        
        # Fetch available Raw Materials
        raw_materials_list = fetch_raw_materials(cursor) if cursor else []
        
        if raw_materials_list:
             item_selection = st.selectbox("Item Name / Number", options=raw_materials_list)
             # Extract item number from "ITEMNMBR - DESC" format
             item = item_selection.split(" - ")[0].strip() if item_selection else ""
        else:
             # Fallback if DB fetch fails or list empty
             item = st.text_input("Item Name / Number", help="Raw material list unavailable")
        
        # Interactive Unit Price Calculation
        pricing_mode = st.radio("Pricing Mode", ["Per Unit", "Per Ton"], horizontal=True)
        
        price_val = 0.0
        if pricing_mode == "Per Ton":
            col_ton, col_lb = st.columns(2)
            with col_ton:
                ton_price = st.number_input("Price per Ton ($)", min_value=0.0, step=1.0, format="%.2f")
            with col_lb:
                # User requested dividing by 500
                calculated_unit = ton_price / 500.0 if ton_price else 0.0
                st.metric("Price per Pound ($)", f"${calculated_unit:,.2f}")
                price_val = calculated_unit
        else:
            price_val = st.number_input("Unit Price ($)", min_value=0.0, step=0.01, format="%.2f")

        valid_until = st.date_input("Price Valid Until", min_value=datetime.date.today())
        
        # Material Location Input (Auto-populated & Required)
        distance_miles = None
        
        # Pre-fill with Vendor Address if available, but allow editing
        location = st.text_input(
            "Material Location / Address (Required)", 
            value=selected_vendor_address,
            placeholder="e.g. 123 Warehouse Way, Tampa, FL"
        )
            
        if location:
            # Calculate distance to 4206 Business Ln, Plant City, FL (28.0186, -82.1129)
            try:
                from geopy.geocoders import Nominatim
                from geopy.distance import geodesic
                
                geolocator = Nominatim(user_agent="cdi_vendor_portal_v1")
                # Use a timeout to prevent hanging
                loc_data = geolocator.geocode(location, timeout=2)
                
                if loc_data:
                    origin_coords = (loc_data.latitude, loc_data.longitude)
                    dest_coords = (28.0186, -82.1129) # 4206 Business Ln, Plant City, FL
                    
                    dist = geodesic(origin_coords, dest_coords).miles
                    distance_miles = round(dist, 1)
                    st.caption(f"📍 Estimated Distance to Plant City: **{distance_miles} miles**")
                else:
                    st.caption("⚠️ Could not locate address for distance calculation.")
            except ImportError:
                    st.warning("Geopy not installed. Distance calculation unavailable.")
            except Exception as e:
                    st.caption(f"⚠️ Distance check failed: {e}")

        # Lead Time Input
        lead_time = st.number_input("Lead Time (Days)", min_value=0, step=1, help="Number of days to produce/deliver material.")

        # Packaging Configuration
        packaging_opts = ["Bulk (Tanker)", "Bulk (Dump/Hopper)", "275gal Tote", "55gal Drum", "50lb Bag", "Super Sack", "Other"]
        packaging = st.selectbox("Packaging Type", packaging_opts)

        # Document Uploads
        st.markdown("### Safety & Compliance")
        col_sds, col_coa = st.columns(2)
        with col_sds:
            sds_file = st.file_uploader("Upload SDS (PDF)", type=["pdf"])
        with col_coa:
            coa_file = st.file_uploader("Upload COA (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

        notes = st.text_area("Additional Notes (MOQ, Lead Time, etc.)")
        
        submitted = st.button("Submit Quote", type="primary")
        if submitted:
            if not item:
                st.error("Please select a valid item.")
            elif price_val <= 0:
                st.error("Please provide a valid price.")
            elif not location:
                 st.error("Material Location is required.")
            else:
                # We store the selected ID if available, or just the user string if not
                save_vendor = selected_vendor_id or st.session_state.user
                save_quote(save_vendor, item, price_val, valid_until, notes, location, distance_miles, lead_time, packaging, sds_file, coa_file)
                st.success(f"Quote for {item} submitted for {vendor_identity}!")
    
    with col2:
        st.subheader("History")
        if os.path.exists(VENDOR_QUOTES_FILE):
            data = []
            try:
                with open(VENDOR_QUOTES_FILE, "r") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            except Exception as e:
                st.error(f"Error reading history: {e}")
            
            if data:
                df = pd.DataFrame(data)
                
                # Global Exclusion Filter
                if not df.empty:
                    df = df[~df['item'].str.startswith("REC-")]
                    df = df[~df['item'].str.startswith("PMX")]
                
                # Filter for this vendor
                my_quotes = df[df.get("vendor") == st.session_state.user]
                
                if not my_quotes.empty:
                    display_cols = ["item", "price", "valid_until", "submitted_at"]
                    st.dataframe(
                        my_quotes[display_cols].sort_values("submitted_at", ascending=False),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("No quotes found for your account.")
            else:
                st.info("No quotes in system.")
        else:
            st.info("No quote history available.")
