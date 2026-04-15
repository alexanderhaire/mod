import pyodbc
from secrets_loader import build_connection_string
import datetime

def check_daily_status_dist():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        target_date = datetime.datetime(2026, 2, 18)
        end_date = target_date + datetime.timedelta(days=1)
        
        print(f"--- Status Distribution for {target_date.date()} ---")
        query = """
            SELECT MANUFACTUREORDERST_I, COUNT(*) 
            FROM WO010032 
            WHERE PSTGDATE >= ? AND PSTGDATE < ?
            GROUP BY MANUFACTUREORDERST_I
        """
        cursor.execute(query, target_date, end_date)
        for r in cursor.fetchall():
            print(f"Status {r[0]}: {r[1]} MOs")
            
        # Also list a few samples of each status
        print("\n--- Samples ---")
        query = """
            SELECT TOP 5 MANUFACTUREORDER_I, MANUFACTUREORDERST_I, DSCRIPTN
            FROM WO010032 
            WHERE PSTGDATE >= ? AND PSTGDATE < ?
            ORDER BY MANUFACTUREORDERST_I
        """
        cursor.execute(query, target_date, end_date)
        for r in cursor.fetchall():
            print(f"{r.MANUFACTUREORDER_I.strip()} | Status {r.MANUFACTUREORDERST_I} | {r.DSCRIPTN.strip()}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_daily_status_dist()
