import datetime
import pyodbc
from secrets_loader import build_connection_string

def debug_query():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        target_date = datetime.date(2026, 2, 18)
        start_dt = datetime.datetime.combine(target_date, datetime.time.min)
        end_dt = start_dt + datetime.timedelta(days=1)
        
        print(f"Target Date: {target_date}")
        print(f"Start DT: {start_dt}")
        print(f"End DT: {end_dt}")
        
        query = """
            SELECT 
                w.MANUFACTUREORDER_I,
                w.PSTGDATE
            FROM WO010032 w
            WHERE w.PSTGDATE >= ? AND w.PSTGDATE < ?
        """
        params = [start_dt, end_dt]
        
        print(f"Executing query with params: {params}")
        cursor.execute(query, params)
        rows = cursor.fetchall()
        print(f"Rows found: {len(rows)}")
        for r in rows:
            print(f"  {r.MANUFACTUREORDER_I}, {r.PSTGDATE}")
            
        print("\n--- Trying String Params ---")
        str_params = [start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')]
        print(f"Params: {str_params}")
        cursor.execute(query, str_params)
        rows = cursor.fetchall()
        print(f"Rows found: {len(rows)}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_query()
