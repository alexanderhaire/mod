import pyodbc
from secrets_loader import build_connection_string

def find_nitrogen_column():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Searching for columns matching 'Nitrogen' in INFORMATION_SCHEMA...")
        
        query = """
        SELECT TABLE_NAME, COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE COLUMN_NAME LIKE '%Nitrogen%' 
           OR COLUMN_NAME LIKE '%NIT%'
        ORDER BY TABLE_NAME
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if rows:
            print(f"Found {len(rows)} matches:")
            for row in rows:
                print(f"Table: {row.TABLE_NAME}, Column: {row.COLUMN_NAME}")
        else:
            print("No columns found matching 'Nitrogen' or 'NIT'.")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_nitrogen_column()
