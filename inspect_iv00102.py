import pyodbc
from secrets_loader import build_connection_string

def inspect_iv00102():
    conn_str, _, _, _ = build_connection_string()
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TOP 1 * FROM IV00102")
            columns = [column[0] for column in cursor.description]
            print("IV00102 Columns:")
            for col in columns:
                print(col)
                
            # Check a few rows specifically for the fields we care about
            print("\nSample Data (First 3 rows):")
            cursor.execute("SELECT TOP 3 ITEMNMBR, LOCNCODE, QTYONHND, ATYALLOC, ORDRPNTQTY, DEX_ROW_TS FROM IV00102 WHERE ORDRPNTQTY > 0")
            rows = cursor.fetchall()
            for row in rows:
                print(row)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_iv00102()
