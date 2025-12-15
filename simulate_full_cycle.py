
import pyodbc
import json
import datetime
import random
import sys
from secrets_loader import build_connection_string
from market_insights import get_priority_raw_materials, calculate_inventory_runway

VENDOR_QUOTES = "vendor_quotes.jsonl"
BROKER_QUOTES = "broker_quotes.jsonl"

def get_db_connection():
    conn_str, _, _, _ = build_connection_string()
    return pyodbc.connect(conn_str)

def simulate_flow():
    print(">> SIMULATING END-TO-END PROCUREMENT FLOW")
    
    # 1. Identify Target
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        print("[1] Scanning for Critical Demand (Runway < 90 Days)...")
        
        df = get_priority_raw_materials(cursor, limit=50)
        target_item = "CHEACETIC" # Default fallback
        target_desc = "Glacial Acetic Acid"
        min_days = 999
        
        for _, row in df.iterrows():
            item = row['ITEMNMBR'].strip()
            runway = calculate_inventory_runway(cursor, item)
            days = runway.get('runway_days', 999)
            
            # Prefer something that isn't completely zero to be realistic, but critical
            if days < 90:
                print(f"    Found candidate: {item} ({days:.1f} days)")
                target_item = item
                target_desc = row['ITEMDESC']
                min_days = days
                if days < 30: 
                    target_item = item # Prioritize critical
                    break
        
        print(f" -> SELECTED TARGET: {target_item} ({target_desc})")
        
    except Exception as e:
        print(f"Error accessing DB: {e}")
        return

    # 2. Inject Vendor Quote (Supply)
    print(f"[2] Injecting Vendor Quote for {target_item}...")
    
    # Generate a quote ID hash
    quote_data = {
        "vendor": "TestVendor_RapidSupply",
        "item": target_item,
        "price": 0.85, # Assuming unit price
        "valid_until": (datetime.date.today() + datetime.timedelta(days=7)).isoformat(),
        "notes": "Urgent stock availability. Instantship.",
        "location": "Savannah, GA",
        "distance_miles": 150,
        "lead_time": 2,
        "packaging": "Bulk (Tanker)",
        "submitted_at": datetime.datetime.now().isoformat()
    }
    
    import hashlib
    q_str = json.dumps(quote_data, sort_keys=True)
    quote_id = hashlib.md5(q_str.encode('utf-8')).hexdigest()
    
    with open(VENDOR_QUOTES, "a") as f:
        f.write(json.dumps(quote_data) + "\n")
    print(f" -> Vendor Quote Injected from 'TestVendor_RapidSupply' @ $0.85")

    # 3. Inject Broker Bid (Logistics)
    print(f"[3] Injecting Broker Bid for Quote {quote_id[:8]}...")
    
    broker_data = {
        "broker_id": "TestBroker_EagleLogistics",
        "vendor_quote_id": quote_id,
        "vendor_quote_summary": {
            "item": target_item,
            "material_price": 0.85
        },
        "freight_price": 1200.00,
        "valid_until": (datetime.date.today() + datetime.timedelta(days=3)).isoformat(),
        "notes": "Dedicated stainless tanker ready for pickup.",
        "equipment_type": "Liquid Tanker (Stainless)",
        "submitted_at": datetime.datetime.now().isoformat()
    }
    
    with open(BROKER_QUOTES, "a") as f:
        f.write(json.dumps(broker_data) + "\n")
        
    print(f" -> Broker Bid Injected from 'TestBroker_EagleLogistics' ($1,200 Freight)")
    print("\n>> SIMULATION COMPLETE. CHECK COMMAND CENTER.")

if __name__ == "__main__":
    simulate_flow()
