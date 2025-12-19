import pyodbc
from secrets_loader import build_connection_string

def find_product():
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        cursor.execute("SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITEMDESC LIKE '%Omega%' OR ITEMDESC LIKE '%Protein%'")
        rows = cursor.fetchall()
        
        print(f"Found {len(rows)} matching items:")
        for row in rows:
            print(f"ID: {row.ITEMNMBR}, Desc: {row.ITEMDESC}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_product()
