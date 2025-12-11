import pyodbc
from constants import RAW_MATERIAL_CLASS_CODES
from secrets_loader import build_connection_string

def debug_item(item_number):
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print(f"--- Debugging {item_number} ---")
        
        # 1. Check Class
        cursor.execute("SELECT ITEMNMBR, ITMCLSCD, ITEMTYPE FROM IV00101 WHERE ITEMNMBR=?", item_number)
        row = cursor.fetchone()
        if row:
            print(f"Class: '{row.ITMCLSCD}', Type: {row.ITEMTYPE}")
            is_rm = str(row.ITMCLSCD).strip().upper() in RAW_MATERIAL_CLASS_CODES
            print(f"Is Raw-Material Class? {is_rm}")
        else:
            print("Item not found in IV00101")
            return

        # 2. Check IV30300 (Finished Good History)
        cursor.execute("SELECT COUNT(*) FROM IV30300 WHERE ITEMNMBR=?", item_number)
        iv_count = cursor.fetchone()[0]
        print(f"IV30300 Rows: {iv_count}")

        # 3. Check POP30310 (Raw Material History)
        cursor.execute("SELECT COUNT(*), MAX(h.RECEIPTDATE) FROM POP30310 l JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM WHERE l.ITEMNMBR=?", item_number)
        row = cursor.fetchone()
        pop_count = row[0]
        max_date = row[1]
        print(f"POP30310 Rows: {pop_count}, Last Receipt: {max_date}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_item('PREMIX93927')
    print("\n")
    debug_item('NPKAWAKEN')
