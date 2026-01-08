import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Checking IV00101 columns for status...")
        cursor.execute("SELECT TOP 1 * FROM IV00101")
        columns = [column[0] for column in cursor.description]
        print(columns)
        
        # Check for INACTIVE or ITEMTYPE
        # ITEMTYPE: 1=Sales Inv, 2=Discontinued
        if 'INACTIVE' in columns:
            print("\nINACTIVE column found.")
            cursor.execute("SELECT TOP 5 ITEMNMBR, ITEMDESC FROM IV00101 WHERE INACTIVE = 1")
            print("Inactive Examples:", cursor.fetchall())
            
        if 'ITEMTYPE' in columns:
            print("\nITEMTYPE column found.")
            cursor.execute("SELECT count(*) FROM IV00101 WHERE ITEMTYPE = 2")
            print(f"Discontinued Items Count: {cursor.fetchone()[0]}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
