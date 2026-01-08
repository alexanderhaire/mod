import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("Checking POP30310 (Receipt Line Hist) columns:")
        cursor.execute("SELECT TOP 1 * FROM POP30310")
        print([c[0] for c in cursor.description])

        print("\nChecking POP30100 (PO Hist Header) columns:")
        cursor.execute("SELECT TOP 1 * FROM POP30100")
        print([c[0] for c in cursor.description])
            
    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
