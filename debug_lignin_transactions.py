
import pyodbc
from secrets_loader import build_connection_string

def inspect_lignin():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # 1. Find items matching LIGNIN
        print("Finding items matching 'LIGNIN'...")
        cursor.execute("SELECT ITEMNMBR, ITEMDESC, ITMCLSCD FROM IV00101 WHERE ITEMDESC LIKE '%LIGNIN%'")
        items = cursor.fetchall()
        
        for item in items:
            item_nmbr = item.ITEMNMBR.strip()
            print(f"\nItem: {item_nmbr} - {item.ITEMDESC} (Class: {item.ITMCLSCD})")
            
            # 2. Check Inventory Transactions (IV30300) for this item
            print("  Transactions (Last 20):")
            cursor.execute("""
                SELECT TOP 20 
                    t.DOCTYPE, 
                    h.IVDOCTYP,
                    t.TRXQTY,
                    h.DOCDATE
                FROM IV30300 t
                JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
                WHERE t.ITEMNMBR = ?
                ORDER BY h.DOCDATE DESC
            """, item_nmbr)
            txns = cursor.fetchall()
            if not txns:
                print("    No transactions found.")
            else:
                for t in txns:
                    print(f"    Date: {t.DOCDATE}, DocType: {t.DOCTYPE}, IVDocType: {t.IVDOCTYP}, Qty: {t.TRXQTY}")
                    
            # 3. Check specific Usage Query behavior
            print("  Current Usage Query Result:")
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN t.TRXQTY < 0 AND h.IVDOCTYP = 1 THEN -t.TRXQTY ELSE 0 END) as UsageQty
                FROM IV30300 t
                JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
                WHERE t.ITEMNMBR = ?
            """, item_nmbr)
            row = cursor.fetchone()
            print(f"    Total Calculated Usage: {row[0] if row else 0}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_lignin()
