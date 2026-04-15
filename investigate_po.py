import pyodbc
from secrets_loader import build_connection_string

def investigate():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Check POP10110 (Open)
        print("Searching POP10110 (Open)...")
        query_open = """
        SELECT 
            L.PONUMBER, L.ITEMNMBR, L.UNITCOST, L.EXTDCOST, 
            H.USER2ENT, H.CREATDDT, H.MODIFDT, H.VENDNAME, H.REVISION_NUMBER
        FROM POP10110 L
        JOIN POP10100 H ON L.PONUMBER = H.PONUMBER
        WHERE L.ITEMNMBR = 'NO3CA' 
        AND (L.UNITCOST > 0.114 AND L.UNITCOST < 0.116)
        ORDER BY H.MODIFDT DESC
        """
        try:
            cursor.execute(query_open)
            rows_open = cursor.fetchall()
            for row in rows_open:
                print(f"OPEN PO: {row.PONUMBER} | Cost: {row.UNITCOST} | User2Ent: {row.USER2ENT} | Created: {row.CREATDDT} | Modified: {row.MODIFDT} | Rev: {row.REVISION_NUMBER}")
        except Exception as e:
            print(f"Error querying OPEN POs: {e}")

        # Check POP30110 (History)
        print("\nSearching POP30110 (History)...")
        query_hist = """
        SELECT 
            L.PONUMBER, L.ITEMNMBR, L.UNITCOST, L.EXTDCOST, 
            H.USER2ENT, H.CREATDDT, H.MODIFDT, H.VENDNAME, H.REVISION_NUMBER
        FROM POP30110 L
        JOIN POP30100 H ON L.PONUMBER = H.PONUMBER
        WHERE L.ITEMNMBR = 'NO3CA'
        AND (L.UNITCOST > 0.114 AND L.UNITCOST < 0.116)
        ORDER BY H.MODIFDT DESC
        """
        try:
            cursor.execute(query_hist)
            rows_hist = cursor.fetchall()
            for row in rows_hist:
                print(f"HIST PO: {row.PONUMBER} | Cost: {row.UNITCOST} | User2Ent: {row.USER2ENT} | Created: {row.CREATDDT} | Modified: {row.MODIFDT} | Rev: {row.REVISION_NUMBER}")
        except Exception as e:
            print(f"Error querying HIST POs: {e}")

    except Exception as e:
        print(f"Error connecting: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    investigate()
