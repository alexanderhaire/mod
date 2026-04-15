import pyodbc
from secrets_loader import build_connection_string

def investigate():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        target_price = 0.11482
        tolerance = 0.00005

        # Check POP10110 (Open PO)
        print("Searching POP10110 (Open PO) for ~0.11482...")
        query_open = f"""
        SELECT 
            L.PONUMBER, L.ITEMNMBR, L.UNITCOST, 
            H.USER2ENT, H.CREATDDT, H.MODIFDT, H.VENDNAME
        FROM POP10110 L
        JOIN POP10100 H ON L.PONUMBER = H.PONUMBER
        WHERE L.ITEMNMBR = 'NO3CA' 
        AND ABS(L.UNITCOST - {target_price}) < {tolerance}
        ORDER BY H.MODIFDT DESC
        """
        cursor.execute(query_open)
        rows = cursor.fetchall()
        for row in rows:
            print(f"OPEN PO HIT: {row.PONUMBER} | Cost: {row.UNITCOST} | User: {row.USER2ENT} | Modified: {row.MODIFDT}")

        # Check POP30110 (History PO)
        print("\nSearching POP30110 (History PO) for ~0.11482...")
        query_hist_po = f"""
        SELECT 
            L.PONUMBER, L.ITEMNMBR, L.UNITCOST, 
            H.USER2ENT, H.CREATDDT, H.MODIFDT, H.VENDNAME
        FROM POP30110 L
        JOIN POP30100 H ON L.PONUMBER = H.PONUMBER
        WHERE L.ITEMNMBR = 'NO3CA' 
        AND ABS(L.UNITCOST - {target_price}) < {tolerance}
        ORDER BY H.MODIFDT DESC
        """
        cursor.execute(query_hist_po)
        rows = cursor.fetchall()
        for row in rows:
            print(f"HIST PO HIT: {row.PONUMBER} | Cost: {row.UNITCOST} | User: {row.USER2ENT} | Modified: {row.MODIFDT}")

        # Check POP30310 (Receipt History)
        print("\nSearching POP30310 (Receipt History) for ~0.11482...")
        # Note: POP30310 joins to POP30300 on POPRCTNM
        query_rcpt = f"""
        SELECT 
            L.POPRCTNM, L.PONUMBER, L.ITEMNMBR, L.UNITCOST, 
            H.USER2ENT, H.CREATDDT, H.MODIFDT, H.VENDNAME
        FROM POP30310 L
        JOIN POP30300 H ON L.POPRCTNM = H.POPRCTNM
        WHERE L.ITEMNMBR = 'NO3CA' 
        AND ABS(L.UNITCOST - {target_price}) < {tolerance}
        ORDER BY H.MODIFDT DESC
        """
        try:
            cursor.execute(query_rcpt)
            rows = cursor.fetchall()
            for row in rows:
                print(f"RECEIPT HIT: {row.POPRCTNM} (PO {row.PONUMBER}) | Cost: {row.UNITCOST} | User: {row.USER2ENT} | Modified: {row.MODIFDT}")
        except Exception as e:
            print(f"Error querying RECEIPTS: {e}")

    except Exception as e:
        print(f"Error connecting: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    investigate()
