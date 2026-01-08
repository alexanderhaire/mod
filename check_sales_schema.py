import pyodbc
from secrets_loader import build_connection_string

def check_sales_tables():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        tables = ["RM00101", "SOP30200"]
        
        for table in tables:
            print(f"\n--- Checking {table} ---")
            try:
                cursor.execute(f"SELECT TOP 1 * FROM {table}")
                columns = [column[0] for column in cursor.description]
                print(f"Columns: {', '.join(columns)}")
                
                # Sample data
                if table == "SOP30200":
                     # Get SOPTYPE=3 (Invoice) typically
                     cursor.execute(f"SELECT TOP 3 DOCDATE, CUSTNMBR, SUB_TOTAL, DOCAMNT FROM {table} WHERE SOPTYPE=3 ORDER BY DOCDATE DESC")
                     print("Sample Sale (Inv):", cursor.fetchall())
                elif table == "RM00101":
                     cursor.execute(f"SELECT TOP 3 CUSTNMBR, CUSTNAME FROM {table}")
                     print("Sample Customers:", cursor.fetchall())

            except Exception as e:
                print(f"Error checking {table}: {e}")

        conn.close()
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    check_sales_tables()
