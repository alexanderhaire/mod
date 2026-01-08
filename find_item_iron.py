import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        search_term = "%iron%hedta%"
        print(f"Searching for items matching: {search_term}")
        
        query = """
        SELECT TOP 5 ITEMNMBR, ITEMDESC 
        FROM IV00101 
        WHERE ITEMDESC LIKE ?
        """
        cursor.execute(query, (search_term,))
        
        rows = cursor.fetchall()
        if not rows:
            print("No matches found. Trying broader search 'iron%'.")
            cursor.execute("SELECT TOP 5 ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITEMDESC LIKE '%iron%'")
            rows = cursor.fetchall()
            
        for row in rows:
            print(f"Found: {row.ITEMNMBR.strip()} - {row.ITEMDESC.strip()}")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
