import pyodbc
from secrets_loader import build_connection_string
import datetime

def check_inventory_218():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        target_date = datetime.datetime(2026, 2, 18)
        end_date = target_date + datetime.timedelta(days=1)
        
        print(f"--- Checking IV30200 (Unposted) for {target_date.date()} ---")
        query = "SELECT DOCNUMBR, IVDOCTYP, BACHNUMB, DOCDATE FROM IV30200 WHERE DOCDATE >= ? AND DOCDATE < ?"
        cursor.execute(query, target_date, end_date)
        rows = cursor.fetchall()
        if rows:
            for r in rows:
                print(f"  {r.DOCNUMBR.strip()} | Type {r.IVDOCTYP} | {r.BACHNUMB.strip()}")
        else:
            print("  No Unposted Transactions found.")

        print(f"\n--- Checking IV30300 (Posted History) for {target_date.date()} ---")
        query = """
            SELECT TOP 20 DOCNUMBR, DOCTYPE, ITEMNMBR, TRXQTY, DOCDATE 
            FROM IV30300 
            WHERE DOCDATE >= ? AND DOCDATE < ?
            ORDER BY DOCNUMBR
        """
        cursor.execute(query, target_date, end_date)
        rows = cursor.fetchall()
        if rows:
            for r in rows:
                print(f"  {r.DOCNUMBR.strip()} | {r.ITEMNMBR.strip()} | {r.TRXQTY}")
        else:
            print("  No Posted Inventory Transactions found.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_inventory_218()
