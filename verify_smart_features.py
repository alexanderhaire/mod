
import pyodbc
import pandas as pd
import numpy as np
from secrets_loader import build_connection_string
from procurement_ml import ProcurementFeatureBuilder
from inventory_queries import fetch_parent_items_for_component
import datetime

def main():
    print("="*60)
    print("SMART FEATURE VERIFICATION SCANNER")
    print("="*60)

    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        print("[OK] DB Connected")
    except Exception as e:
        print(f"[FAIL] {e}")
        return

    builder = ProcurementFeatureBuilder(cursor)
    
    # 1. Get a list of Candidate Items (Raw Materials likely to have volatility or parents)
    print("\nScanning specific known items for Volatility & BOM...")
    # Items found via debug_volatility.py to have data
    item_list = ['NPK15012', 'MRCCCN02', 'THIOCA', 'SO4MN32'] 
    
    items = item_list
    print(f"-> Checking {len(items)} specific candidates.")

    found_volatility = False
    found_bom = False

    print("\n--- Checking Lead Time Volatility ---")
    for item in items:
        # We access the internal method to check the raw calculation
        vendor_info = builder._get_vendor_info(item)
        if not vendor_info:
            continue
            
        vol = vendor_info.get('lead_time_volatility', 0.0)
        avg_gap = vendor_info.get('actual_lead_time', 0)
        
        if vol > 0:
            print(f"[FOUND] Item {item}: Volatility = {vol:.2f} days (Avg Lead: {avg_gap} days)")
            found_volatility = True
            if found_bom: break # Stop if we found both, else continue
    
    if not found_volatility:
        print("[WARN] No items with >0 volatility found in sample. (This might be normal if data is clean/synthetic)")

    print("\n--- Checking BOM Parent Trends ---")
    # Reset loop or pick new items
    for item in items:
        parents, _ = fetch_parent_items_for_component(cursor, item)
        if parents:
            print(f"[FOUND] Item {item} is used in {len(parents)} parents.")
            for p in parents:
                print(f"   -> Used in: {p.ParentItem.strip()} (Qty: {p.QtyPerParent:.4f})")
                
                # Check trend logic
                features = builder.build_features(item)
                print(f"   -> Calculated Parent Trend Feature: {features.parent_demand_trend:.6f}")
                
            found_bom = True
            break
            
    if not found_bom:
        print("[WARN] No items found that are components of other items (Reverse BOM empty for sample).")

if __name__ == "__main__":
    main()
