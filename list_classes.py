
import pyodbc
from secrets_loader import build_connection_string

def list_classes():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("--- Unique Class Codes ---")
        cursor.execute("SELECT DISTINCT ITMCLSCD FROM IV00101 ORDER BY ITMCLSCD")
        rows = cursor.fetchall()
        for r in rows:
            print(r.ITMCLSCD)
            
        print("\n--- Searching for 'EPO' in Item Description ---")
        cursor.execute("SELECT TOP 20 ITEMNMBR, ITEMDESC, ITMCLSCD FROM IV00101 WHERE ITEMDESC LIKE '%EPO%'")
        rows = cursor.fetchall()
        for r in rows:
            print(f"Item: {r.ITEMNMBR}, Desc: {r.ITEMDESC}, Class: {r.ITMCLSCD}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    list_classes()
