
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Searching for receipt with UNITCOST approx 0.5875...")
    # Using range for float comparison
    cursor.execute("""
        SELECT TOP 5
            l.ITEMNMBR,
            l.POPRCTNM,
            l.UNITCOST,
            l.EXTDCOST,
            h.ORFRTAMT,
            h.SUBTOTAL,
            h.RECEIPTDATE
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE l.UNITCOST >= 0.587 AND l.UNITCOST <= 0.588
        ORDER BY h.RECEIPTDATE DESC
    """)
    
    rows = cursor.fetchall()
    if rows:
        print(f"Found {len(rows)} matching receipts:")
        for row in rows:
            print(f"Item: {row.ITEMNMBR}, Receipt: {row.POPRCTNM}, Cost: {row.UNITCOST}, Freight: {row.ORFRTAMT}, Subtotal: {row.SUBTOTAL}, Date: {row.RECEIPTDATE}")
    else:
        print("No matching receipts found.")
        
    conn.close()
except Exception as e:
    print(f"Error: {e}")
