import pyodbc
from secrets_loader import build_connection_string

def check_pos():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        po_numbers = ["431-8159", "431-8189", "431-8190"]
        joined_pos = "'" + "','".join(po_numbers) + "'"

        print(f"Checking POs: {joined_pos}")

        # Open POs
        print("\n--- OPEN POs (POP10110) ---")
        q_open = f"""
        SELECT L.PONUMBER, L.ITEMNMBR, L.UNITCOST, H.USER2ENT, H.MODIFDT
        FROM POP10110 L JOIN POP10100 H ON L.PONUMBER = H.PONUMBER
        WHERE L.PONUMBER IN ({joined_pos})
        """
        cursor.execute(q_open)
        for r in cursor.fetchall():
            print(f"PO: {r.PONUMBER} | Item: {r.ITEMNMBR} | Cost: {r.UNITCOST} | User: {r.USER2ENT}")

        # History POs
        print("\n--- HISTORY POs (POP30110) ---")
        q_hist = f"""
        SELECT L.PONUMBER, L.ITEMNMBR, L.UNITCOST, H.USER2ENT, H.MODIFDT
        FROM POP30110 L JOIN POP30100 H ON L.PONUMBER = H.PONUMBER
        WHERE L.PONUMBER IN ({joined_pos})
        """
        cursor.execute(q_hist)
        for r in cursor.fetchall():
            print(f"PO: {r.PONUMBER} | Item: {r.ITEMNMBR} | Cost: {r.UNITCOST} | User: {r.USER2ENT}")

        # Receipts
        print("\n--- RECEIPTS (POP30310) ---")
        q_rcpt = f"""
        SELECT L.POPRCTNM, L.PONUMBER, L.ITEMNMBR, L.UNITCOST, H.USER2ENT, H.MODIFDT
        FROM POP30310 L JOIN POP30300 H ON L.POPRCTNM = H.POPRCTNM
        WHERE L.PONUMBER IN ({joined_pos})
        """
        cursor.execute(q_rcpt)
        for r in cursor.fetchall():
            print(f"RCPT: {r.POPRCTNM} (PO {r.PONUMBER}) | Item: {r.ITEMNMBR} | Cost: {r.UNITCOST} | User: {r.USER2ENT}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_pos()
