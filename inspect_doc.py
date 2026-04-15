import pyodbc
from secrets_loader import build_connection_string

def inspect_doc():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Bad?
        doc_bad = '000148059'
        
        print(f"\n=== Inspecting {doc_bad} ===")
        print(f"--- IV30300 (Posted) ---")
        cursor.execute("SELECT DOCNUMBR, ITEMNMBR, TRXQTY FROM IV30300 WHERE DOCNUMBR = ?", doc_bad)
        rows = cursor.fetchall()
        if rows:
            print(f"  Found {len(rows)} lines in IV30300.")
            for r in rows:
                print(f"    Item: {r.ITEMNMBR} | Qty: {r.TRXQTY}")
        else:
            print("  Not found in IV30300.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_doc()
