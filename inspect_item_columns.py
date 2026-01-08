import pyodbc
from secrets_loader import build_connection_string

def inspect_columns():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        table_name = "IV00101"
        print(f"Inspecting columns for {table_name}...")
        
        cursor.execute(f"SELECT TOP 1 * FROM {table_name}")
        columns = [column[0] for column in cursor.description]
        
        print(f"Total Columns: {len(columns)}")
        print(f"Columns: {', '.join(columns)}")
        
        # Check specific patterns
        potential_matches = [col for col in columns if "NIT" in col or "USR" in col or "CAT" in col or "DEF" in col or "CHECK" in col or "FLAG" in col]
        print(f"\nPotential matches for Nitrogen/User fields: {potential_matches}")
        
        # Dump a row to see values
        print("\nFirst row values for potential matches:")
        row = cursor.fetchone()
        if row:
            for col in potential_matches:
                 # Find index
                idx = columns.index(col)
                print(f"{col}: {row[idx]}")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_columns()
