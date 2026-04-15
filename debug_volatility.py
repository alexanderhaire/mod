
import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("\n--- Checking for ANY item with volatility > 0 ---")
    cursor.execute("""
        SELECT TOP 10 
            iv.ITEMNMBR,
            COUNT(*) as ReceiptCount,
            STDEV(DATEDIFF(day, po.DOCDATE, h.RECEIPTDATE)) as Val
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        JOIN POP30100 po ON l.PONUMBER = po.PONUMBER
        JOIN IV00101 iv ON l.ITEMNMBR = iv.ITEMNMBR
        WHERE l.PONUMBER <> '' 
          AND h.RECEIPTDATE >= po.DOCDATE
          AND h.RECEIPTDATE > DATEADD(year, -10, GETDATE()) -- Look back further
        GROUP BY iv.ITEMNMBR
        HAVING COUNT(*) > 1 AND STDEV(DATEDIFF(day, po.DOCDATE, h.RECEIPTDATE)) > 0
        ORDER BY Val DESC
    """)
    
    rows = cursor.fetchall()
    if rows:
        print(f"[SUCCESS] Found {len(rows)} volatile items.")
        for r in rows:
            print(f"  Item: {r.ITEMNMBR} | Count: {r.ReceiptCount} | Volatility: {r.Val:.4f}")
    else:
        print("[FAIL] No items with standard deviation > 0 found in the entire database.")
        
    print("\n--- Checking Parent Usage Logic ---")
    # Check if we can find a parent with actual usage slope
    cursor.execute("""
        SELECT TOP 5 ITEMNMBR, COUNT(*) as SalesCount 
        FROM IV30300 
        WHERE DOCDATE > DATEADD(year, -1, GETDATE()) 
          AND TRXQTY <> 0
        GROUP BY ITEMNMBR 
        HAVING COUNT(*) > 5
    """)
    sales = cursor.fetchall()
    print(f"Found {len(sales)} items with active transaction history.")
    for s in sales:
        print(f"  Parent candidate: {s.ITEMNMBR} ({s.SalesCount} txns)")

if __name__ == "__main__":
    main()
