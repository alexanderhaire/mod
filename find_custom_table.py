import pyodbc
from secrets_loader import build_connection_string

def find_733():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Searching for string '733' in data...")
        
        # specific focus on likely tables first to save time
        # Tables starting with IV, EXT, POP, SOP
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
        tables = [row.TABLE_NAME for row in cursor.fetchall()]
        
        candidates = []
        for table in tables:
            # Skip huge transaction tables if possible, but need to be thorough
            if "HIST" in table: continue 
            
            try:
                # check if table has any text/int column
                cursor.execute(f"SELECT TOP 1 * FROM {table}")
                cols = [c[0] for c in cursor.description]
                
                # construct naive query
                # logical OR across all columns
                where_clause = " OR ".join([f"CAST([{col}] AS VARCHAR(MAX)) = '733'" for col in cols])
                
                query = f"SELECT TOP 1 * FROM {table} WHERE {where_clause}"
                cursor.execute(query)
                if cursor.fetchone():
                    print(f"MATCH FOUND IN TABLE: {table}")
                    candidates.append(table)
            except Exception as e:
                pass # ignore errors on weird types
                
        print(f"Candidates: {candidates}")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_733()
