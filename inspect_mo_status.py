import pyodbc
from secrets_loader import build_connection_string
import datetime

def inspect_mo_status():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # MOs that appeared in the report (sample)
        mos = [
            'X2602-0018', 'X2602-0045', 'X2602-0065', 
            'C2602-0054', 'C2602-0055', 'C2602-0058'
        ]
        
        print(f"--- Inspecting Status for {len(mos)} Sample MOs ---")
        placeholders = ",".join("?" for _ in mos)
        query = f"""
            SELECT 
                MANUFACTUREORDER_I, 
                MANUFACTUREORDERST_I, 
                PSTGDATE, 
                ENDQTY_I 
            FROM WO010032 
            WHERE MANUFACTUREORDER_I IN ({placeholders})
        """
        cursor.execute(query, mos)
        rows = cursor.fetchall()
        
        print(f"{'MO Number':<15} | {'Status':<6} | {'Posting Date':<10} | {'Qty':<10}")
        print("-" * 50)
        for r in rows:
            print(f"{r.MANUFACTUREORDER_I.strip():<15} | {r.MANUFACTUREORDERST_I:<6} | {r.PSTGDATE} | {r.ENDQTY_I}")
            
        # Also check what OTHER statuses exist in the table
        print("\n--- Distribution of MO Statuses in WO010032 ---")
        cursor.execute("SELECT MANUFACTUREORDERST_I, COUNT(*) FROM WO010032 GROUP BY MANUFACTUREORDERST_I")
        for r in cursor.fetchall():
            print(f"Status {r[0]}: {r[1]} records")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_mo_status()
