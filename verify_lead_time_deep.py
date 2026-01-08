import pyodbc
from secrets_loader import build_connection_string
from procurement_ml import ProcurementMLOptimizer

def main():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Find an item with recent history linked to a PO
        print("Finding item with PO history...")
        query = """
        SELECT TOP 1 l.ITEMNMBR
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        JOIN POP30100 po ON l.PONUMBER = po.PONUMBER
        WHERE h.RECEIPTDATE > DATEADD(year, -2, GETDATE())
          AND l.PONUMBER <> ''
        """
        cursor.execute(query)
        result = cursor.fetchone()
        
        if not result:
            print("No items found with linked PO history!")
            return
            
        item = result[0].strip()
        print(f"Found item: {item}")
        
        optimizer = ProcurementMLOptimizer(cursor)
        df = optimizer.get_batch_recommendations([item])
        
        for _, row in df.iterrows():
            rec = optimizer.get_buy_recommendation(row['ItemNumber'])
            feat = rec['features']
            print(f"\nItem: {row['ItemNumber']}")
            print(f"  Plan LeadTime: {feat.get('planning_lead_time')}")
            print(f"  ACTUAL LeadTime: {feat.get('actual_lead_time')}")
            print(f"  Effective Used: {feat.get('lead_time')}")

    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
