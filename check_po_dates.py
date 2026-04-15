import pyodbc
from db_pool import get_connection

def check_po_dates():
    print("--- Checking PO Date Ranges ---")
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Check History (POP30100)
        try:
            cursor.execute("SELECT MIN(DOCDATE), MAX(DOCDATE), COUNT(*) FROM POP30100")
            min_d, max_d, count = cursor.fetchone()
            print(f"History (POP30100): {count} POs, from {min_d} to {max_d}")
        except Exception as e:
            print(f"History Error: {e}")

        # Check Work (POP10100)
        try:
            cursor.execute("SELECT MIN(DOCDATE), MAX(DOCDATE), COUNT(*) FROM POP10100")
            result = cursor.fetchone()
            if result:
                min_d, max_d, count = result
                print(f"Work (POP10100):    {count} POs, from {min_d} to {max_d}")
            else:
                print("Work (POP10100):    Empty")
        except Exception as e:
            print(f"Work Error: {e}")

if __name__ == "__main__":
    check_po_dates()
