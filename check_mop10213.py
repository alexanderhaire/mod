import pyodbc
from secrets_loader import build_connection_string

def check_mop10213():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        mos = ['X2602-0018', 'C2602-0054']
        
        print(f"--- Checking MOP10213 for MOs {mos} ---")
        placeholders = ",".join("?" for _ in mos)
        query = f"SELECT * FROM MOP10213 WHERE MANUFACTUREORDER_I IN ({placeholders})"
        cursor.execute(query, mos)
        rows = cursor.fetchall()
        if rows:
            print(f"Found {len(rows)} rows.")
            columns = [column[0] for column in cursor.description]
            print(columns)
            for r in rows: 
                print(list(r))
        else:
            print("  No matches in MOP10213.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_mop10213()
