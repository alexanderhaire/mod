import pyodbc
from secrets_loader import build_connection_string

def debug_c2602_0159():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        mo = 'C2602-0159'
        print(f"--- Inspecting {mo} ---")
        
        # 1. Check WO010032 Master
        cursor.execute("SELECT MANUFACTUREORDERST_I, PSTGDATE, ENDQTY_I FROM WO010032 WHERE MANUFACTUREORDER_I = ?", mo)
        row = cursor.fetchone()
        if row:
            print(f"Master: Status={row.MANUFACTUREORDERST_I}, Date={row.PSTGDATE}, Qty={row.ENDQTY_I}")
        else:
            print("Master: Not found")
            
        # 2. Check MOP10213 Receipts
        print("\n--- MOP10213 Entries ---")
        cursor.execute("SELECT MANUFACTUREORDER_I, RCPTNMBR, IVDOCNBR, DOCTYPE FROM MOP10213 WHERE MANUFACTUREORDER_I = ?", mo)
        receipts = cursor.fetchall()
        for r in receipts:
            print(f"Rcpt: {r.RCPTNMBR} | Doc: {r.IVDOCNBR} | Type: {r.DOCTYPE}")
            
            # Check IV30300 for this Doc
            cursor.execute("SELECT ITEMNMBR, TRXQTY, DOCDATE FROM IV30300 WHERE DOCNUMBR = ?", r.IVDOCNBR)
            lines = cursor.fetchall()
            for l in lines:
                print(f"  Line: {l.ITEMNMBR.strip()} | Qty: {l.TRXQTY} | Date: {l.DOCDATE}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_c2602_0159()
