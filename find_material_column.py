import pyodbc
from secrets_loader import build_connection_string

def find_material_column():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Searching for columns matching 'Material' in INFORMATION_SCHEMA...")
        
        query = """
        SELECT TABLE_NAME, COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE COLUMN_NAME LIKE '%Material%' 
        ORDER BY TABLE_NAME
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if rows:
            print(f"Found {len(rows)} matches. Top 20 relevant tables:")
            # Filter for likely custom tables or core IV tables
            seen_tables = set()
            for row in rows:
                if row.TABLE_NAME not in seen_tables:
                    print(f"Table: {row.TABLE_NAME}, Column: {row.COLUMN_NAME}")
                    seen_tables.add(row.TABLE_NAME)
        else:
            print("No columns found matching 'Material'.")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_material_column()
