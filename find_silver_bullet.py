import pyodbc
from secrets_loader import build_connection_string

def find_item_and_value():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        target_item = "GOLDCA00"
        target_value = "733"
        print(f"Searching for table containing item '{target_item}' and value '{target_value}'...")
        
        # 1. Get tables with ITEMNMBR
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE COLUMN_NAME = 'ITEMNMBR'
        """)
        tables = [row.TABLE_NAME for row in cursor.fetchall()]
        print(f"Found {len(tables)} tables with ITEMNMBR column.")
        
        candidates = []
        
        for table in tables:
            try:
                # Check if item exists in table
                # Use quoted identifier to handle special characters in table names
                cursor.execute(f"SELECT TOP 1 * FROM \"{table}\" WHERE ITEMNMBR = ?", (target_item,))
                row = cursor.fetchone()
                
                if row:
                    # Item exists, scan row values for 733 or "Nitrogen" column
                    columns = [column[0] for column in cursor.description]
                    
                    found_value = False
                    found_col = False
                    
                    # Check for value 733
                    for col, val in zip(columns, row):
                        if str(val).strip() == target_value:
                            print(f"\n!!! FOUND VALUE !!! Table: {table}, Column: {col} = {val}")
                            candidates.append(table)
                            found_value = True
                            
                    # Check for Nitrogen column
                    for col in columns:
                        if "NITROGEN" in col.upper() or "NIT" in col.upper():
                             # Ignore simple unit/init matches if common
                            if col.upper() not in ["UNITCOST", "UNITPRCE", "INIT"]: 
                                print(f"Found suspicious column in {table}: {col}")
                                
                    if found_value:
                        # Inspect full row
                        print(f"Row data for {table}:")
                        for col, val in zip(columns, row):
                            print(f"  {col}: {val}")
                            
            except Exception as e:
                # print(f"Skipping {table}: {e}")
                pass

        print("\nSearch complete.")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_item_and_value()
