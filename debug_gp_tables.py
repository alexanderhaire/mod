import pyodbc
from secrets_loader import build_connection_string

def inspect_data():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'NO3CA12'
        
        print(f"--- IV00102 Data for {item} ---")
        cursor.execute("SELECT RCRDTYPE, LOCNCODE, QTYONHND, ATYALLOC, QTYONORD FROM IV00102 WHERE ITEMNMBR = ?", item)
        rows = cursor.fetchall()
        for row in rows:
            print(f"Type: {row.RCRDTYPE}, Loc: {row.LOCNCODE}, OnHand: {row.QTYONHND}, Alloc: {row.ATYALLOC}, OnOrd: {row.QTYONORD}")
            
        print(f"\n--- POP10110 Columns ---")
        cursor.execute("SELECT TOP 1 * FROM POP10110")
        col_names = [column[0] for column in cursor.description]
        print(col_names)

        print(f"\n--- POP10110 Data for {item} ---")
        # Try to select relevant columns based on standard GP schema, but print what we find
        # QTYORDER, QTYCANCE, QTYINVCD is standard? Let's check.
        cursor.execute("SELECT PONUMBER, ITEMNMBR, QTYORDER, QTYCANCE FROM POP10110 WHERE ITEMNMBR = ?", item)
        po_rows = cursor.fetchall()
        for r in po_rows:
            print(f"PO: {r.PONUMBER}, QtyOrd: {r.QTYORDER}, QtyCnc: {r.QTYCANCE}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_data()
