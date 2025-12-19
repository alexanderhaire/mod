
import pyodbc
from secrets_loader import build_connection_string

def find_item():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Searching for 'Ascorbic' in Item Descriptions...")
        cursor.execute("SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITEMDESC LIKE '%Ascorbic%'")
        rows = cursor.fetchall()
        
        if not rows:
            print("No matches found in IV00101.")
        else:
            print(f"Found {len(rows)} matches:")
            for row in rows:
                print(f"ID: {row.ITEMNMBR.strip()}, Desc: {row.ITEMDESC.strip()}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_item()
