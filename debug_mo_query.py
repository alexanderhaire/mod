import pyodbc
from secrets_loader import build_connection_string
import datetime

def debug_mo_query():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        target_mo = 'X2602-0018'
        
        print(f"--- Debugging Query for {target_mo} ---")
        query = """
            SELECT 
                w.MANUFACTUREORDER_I, 
                m.IVDOCNBR, 
                iv.DOCNUMBR, 
                iv.ITEMNMBR, 
                w.ITEMNMBR as ExpectedItem,
                iv.TRXQTY 
            FROM WO010032 w
            JOIN MOP10213 m ON w.MANUFACTUREORDER_I = m.MANUFACTUREORDER_I
            JOIN IV30300 iv ON m.IVDOCNBR = iv.DOCNUMBR
            WHERE w.MANUFACTUREORDER_I = ?
        """
        cursor.execute(query, target_mo)
        rows = cursor.fetchall()
        print(f"Rows found: {len(rows)}")
        for r in rows:
            print(f"  Doc: {r.IVDOCNBR} | Item: {r.ITEMNMBR} (Exp: {r.ExpectedItem}) | Qty: {r.TRXQTY}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_mo_query()
