import pyodbc
from secrets_loader import build_connection_string

def check_schema():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Checking IV30300 columns...")
        cursor.execute("SELECT TOP 1 * FROM IV30300")
        columns = [column[0] for column in cursor.description]
        print(f"Columns: {columns}")
        
        if 'DOCDATE' in columns:
            print("DOCDATE is present in IV30300.")
        else:
            print("DOCDATE is NOT present in IV30300. Need to join IV30200.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_schema()
