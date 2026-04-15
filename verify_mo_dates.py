import pyodbc
from secrets_loader import build_connection_string

def verify_mo_dates():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        print("--- Verifying MO Dates and Quantities (WO010032) ---")
        
        # Check recent C-MOs
        query = """
        SELECT TOP 10 
            MANUFACTUREORDER_I, 
            ITEMNMBR, 
            ENDQTY_I, 
            STRTDATE, 
            ENDDATE, 
            PSTGDATE, 
            DSCRIPTN 
        FROM WO010032 
        WHERE MANUFACTUREORDER_I LIKE 'C%' 
        AND PSTGDATE > '2025-01-01'
        ORDER BY PSTGDATE DESC
        """
        cursor.execute(query)
        print("Recent C-MOs:")
        for r in cursor.fetchall(): print(f"  {list(r)}")
        
        # Check recent X-MOs
        query = """
        SELECT TOP 10 
            MANUFACTUREORDER_I, 
            ITEMNMBR, 
            ENDQTY_I, 
            STRTDATE, 
            ENDDATE, 
            PSTGDATE, 
            DSCRIPTN 
        FROM WO010032 
        WHERE MANUFACTUREORDER_I LIKE 'X%' 
        AND PSTGDATE > '2025-01-01'
        ORDER BY PSTGDATE DESC
        """
        cursor.execute(query)
        print("\nRecent X-MOs:")
        for r in cursor.fetchall(): print(f"  {list(r)}")

    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_mo_dates()
