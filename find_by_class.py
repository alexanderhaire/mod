
import pyodbc
from secrets_loader import build_connection_string

def find_by_class():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        classes = ['EPO62', 'MISCINTGR80']
        
        for cls in classes:
            print(f"\n--- Searching for items in Class: {cls} ---")
            cursor.execute("SELECT ITEMNMBR, ITEMDESC, ITMCLSCD FROM IV00101 WHERE ITMCLSCD = ?", cls)
            rows = cursor.fetchall()
            if not rows:
                print("No items found.")
            for r in rows:
                print(f"Item: {r.ITEMNMBR}, Desc: {r.ITEMDESC}")
                
                # Check receipts for this item
                print(f"  Checking receipts for {r.ITEMNMBR} in Dec 2025...")
                cursor.execute("""
                    SELECT h.POPRCTNM, h.RECEIPTDATE, h.VENDORID, l.ACTLSHIP, l.UNITCOST, l.EXTDCOST 
                    FROM POP30310 l
                    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
                    WHERE l.ITEMNMBR = ? AND h.RECEIPTDATE >= '2025-12-01'
                """, r.ITEMNMBR)
                receipts = cursor.fetchall()
                for rcpt in receipts:
                    print(f"    RCT: {rcpt.POPRCTNM}, Date: {rcpt.RECEIPTDATE}, Qty: {rcpt.ACTLSHIP}, Cost: {rcpt.UNITCOST}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    find_by_class()
