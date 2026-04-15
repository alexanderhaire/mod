import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def verify_perpetual_alignment():
    """
    Verify that the perpetual inventory report correctly calculates:
    1. Quantity as of the target date
    2. Cost as of the target date (last transaction cost on or before date)
    """
    target_date = date(2025, 9, 30)
    test_items = ['ZZ2.5GALF', 'GOLDFE02', 'NPKKTS', 'ZZ55GAL', 'ZZ30GAL']
    
    print(f"=" * 70)
    print(f"PERPETUAL INVENTORY VERIFICATION - As of {target_date}")
    print(f"=" * 70)
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    results = []
    
    for item in test_items:
        print(f"\n--- {item} ---")
        
        # 1. Get Current Qty and Cost
        q_current = """
        SELECT T1.CURRCOST, T2.QTYONHND as CurrentQty
        FROM IV00101 T1
        JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
        WHERE T1.ITEMNMBR = ? AND T2.LOCNCODE = 'MAIN'
        """
        current = pd.read_sql(q_current, conn, params=[item])
        current_qty = current['CurrentQty'].iloc[0] if not current.empty else 0
        current_cost = current['CURRCOST'].iloc[0] if not current.empty else 0
        
        # 2. Get Qty Change after target date
        q_change = """
        SELECT SUM(TRXQTY) as QtyChange
        FROM IV30300
        WHERE ITEMNMBR = ? AND DOCDATE > ? AND TRXLOCTN = 'MAIN'
        """
        change = pd.read_sql(q_change, conn, params=[item, target_date])
        qty_change = change['QtyChange'].iloc[0] if change['QtyChange'].iloc[0] else 0
        
        # 3. Get Last Cost on or before target date
        q_hist_cost = """
        SELECT TOP 1 UNITCOST, DOCDATE, DOCTYPE
        FROM IV30300
        WHERE ITEMNMBR = ? AND DOCDATE <= ? AND UNITCOST IS NOT NULL AND UNITCOST <> 0
        ORDER BY DOCDATE DESC, DEX_ROW_ID DESC
        """
        hist_cost = pd.read_sql(q_hist_cost, conn, params=[item, target_date])
        
        if hist_cost.empty:
            historical_cost = current_cost  # Fallback
            cost_source = "Fallback (Current)"
            cost_date = "N/A"
        else:
            historical_cost = hist_cost['UNITCOST'].iloc[0]
            cost_source = f"Historical (DocType {hist_cost['DOCTYPE'].iloc[0]})"
            cost_date = str(hist_cost['DOCDATE'].iloc[0])
        
        # Calculate As Of values
        as_of_qty = current_qty - qty_change
        
        print(f"  Current Qty:     {current_qty:,.2f}")
        print(f"  Qty Change After: {qty_change:,.2f}")
        print(f"  As Of Qty:       {as_of_qty:,.2f}")
        print(f"  Current Cost:    ${current_cost:,.4f}")
        print(f"  Historical Cost: ${historical_cost:,.4f} ({cost_source}, Date: {cost_date})")
        print(f"  Cost Match Current? {'YES' if abs(current_cost - historical_cost) < 0.01 else 'NO - DIFFERENT'}")
        
        results.append({
            'Item': item,
            'As Of Qty': as_of_qty,
            'Current Cost': current_cost,
            'Historical Cost': historical_cost,
            'Cost Source': cost_source,
            'Cost Date': cost_date
        })
    
    conn.close()
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    verify_perpetual_alignment()
