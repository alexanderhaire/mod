import pyodbc
from secrets_loader import build_connection_string
from procurement_ml import ProcurementMLOptimizer

def main():
    print("Verifying Lead Time Logic...")
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Conn failed: {e}")
        return

    optimizer = ProcurementMLOptimizer(cursor)
    
    # Get a few items
    try:
        cursor.execute("SELECT TOP 5 ITEMNMBR FROM IV00101 WHERE ITMCLSCD IN ('RAWMATNTB') AND INACTIVE=0")
        items = [r.ITEMNMBR.strip() for r in cursor.fetchall()]
        
        print(f"Testing items: {items}")
        
        df = optimizer.get_batch_recommendations(items)
        if df.empty:
            print("No recs found")
            return
            
        print("\n--- Lead Time Results ---")
        for idx, row in df.iterrows():
            # Get internal features if possible, or infer from output
            # Note: The DF output flattens some things, let's check debug columns if present
            # or just run single get_buy_recommendation for deeper inspection
            
            # Deeper inspection
            rec = optimizer.get_buy_recommendation(row['ItemNumber'])
            feat = rec['features']
            
            print(f"Item: {row['ItemNumber']}")
            print(f"  Plan LeadTime: {feat.get('planning_lead_time')}")
            print(f"  ACTUAL LeadTime: {feat.get('actual_lead_time')}")
            print(f"  Effective Used: {feat.get('lead_time')}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
