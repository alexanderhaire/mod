import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Check IV00103 (Item Vendor) for lead time columns
        print("Checking IV00103 columns...")
        cursor.execute("SELECT TOP 1 * FROM IV00103")
        columns = [column[0] for column in cursor.description]
        print(columns)
        
        # Check distribution
        print("\n--- Lead Time Stats ---")
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN PLANNINGLEADTIME > 0 THEN PLANNINGLEADTIME ELSE NULL END) as AvgPlan,
                COUNT(CASE WHEN PLANNINGLEADTIME > 0 THEN 1 ELSE NULL END) as CountPlan,
                AVG(CASE WHEN AVRGLDTM > 0 THEN AVRGLDTM ELSE NULL END) as AvgActual,
                COUNT(CASE WHEN AVRGLDTM > 0 THEN 1 ELSE NULL END) as CountActual
            FROM IV00103
        """)
        row = cursor.fetchone()
        print(f"Planning Lead Time: Avg={row.AvgPlan}, Count={row.CountPlan}")
        print(f"Average Lead Time: Avg={row.AvgActual}, Count={row.CountActual}")
        
        # Show some examples
        print("\n--- Examples ---")
        cursor.execute("""
            SELECT TOP 5 ITEMNMBR, PLANNINGLEADTIME, AVRGLDTM
            FROM IV00103
            WHERE PLANNINGLEADTIME > 0 OR AVRGLDTM > 0
        """)
        for r in cursor.fetchall():
            print(f"Item: {r.ITEMNMBR.strip()}, Plan: {r.PLANNINGLEADTIME}, Actual: {r.AVRGLDTM}")

    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
