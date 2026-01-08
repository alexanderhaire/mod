import pyodbc
import datetime
import pandas as pd
from secrets_loader import build_connection_string
from procurement_ml import ProcurementMLOptimizer

def main():
    item_num = "EDTAFEHE"
    print(f"Analyzing {item_num}...")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    optimizer = ProcurementMLOptimizer(cursor)
    
    # 1. Check Usage History
    print("\n--- Usage History (Last 3 months) ---")
    usage_df = optimizer.feature_builder._get_usage_history(item_num, datetime.date.today())
    if not usage_df.empty:
        print(usage_df.tail(10))
    else:
        print("No usage found.")

    # 2. Check Recent POs
    print("\n--- Recent POs ---")
    # 1b. Check Annual Usage
    print("\n--- Annual Usage ---")
    cursor.execute("""
        SELECT YEAR(DOCDATE) as Year, SUM(ABS(TRXQTY)) as Qty
        FROM IV30300
        WHERE ITEMNMBR = ?
          AND DOCTYPE IN (1, 5) -- Sales, Adjustments
          AND TRXQTY < 0
        GROUP BY YEAR(DOCDATE)
        ORDER BY YEAR(DOCDATE) DESC
    """, (item_num,))
    
    for row in cursor.fetchall():
        print(f"Year: {row.Year} Usage: {row.Qty:,.0f}")
    
    rows = cursor.fetchall()
    if not rows:
         print("No recent POs found.")
    else:
        for row in rows:
            print(f"PO: {row.POPRCTNM} Date: {row.RECEIPTDATE} Qty: {row.Quantity} Price: ${row.UNITCOST:.2f} Vendor: {row.VENDORID}")

    # 2b. Check Open POs (Work)
    print("\n--- Open POs (Recent) ---")
    cursor.execute("""
        SELECT TOP 3 h.PONUMBER, h.DOCDATE, l.QTYORDER, l.UNITCOST, h.VENDORID
        FROM POP10110 l
        JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
        WHERE l.ITEMNMBR = ?
        ORDER BY h.DOCDATE DESC
    """, (item_num,))
    
    rows = cursor.fetchall()
    if not rows:
        print("No open POs found.")
    else:
        for row in rows:
            print(f"Open PO: {row.PONUMBER} Date: {row.DOCDATE} Qty: {row.QTYORDER:.2f} Price: ${row.UNITCOST:.2f} Vendor: {row.VENDORID}")

    # 3. Check Price History Trend
    print("\n--- Price Trend Check ---")
    price_df = optimizer.feature_builder._get_price_history(item_num, datetime.date.today())
    if not price_df.empty:
        print(f"Current Price: ${price_df.iloc[-1]['UnitCost']:.2f}")
        print("Recent prices:")
        print(price_df.tail(5)[['TransactionDate', 'UnitCost']])
    
    conn.close()

if __name__ == "__main__":
    main()
