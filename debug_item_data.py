import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'MRCCCN02'
        print(f"Checking data for {item}...")
        
        # Check IV00103 (Item Vendor Master)
        print("\nIV00103 Entires:")
        cursor.execute("SELECT ITEMNMBR, VENDORID, PLANNINGLEADTIME FROM IV00103 WHERE ITEMNMBR = ?", item)
        rows = cursor.fetchall()
        for r in rows:
            print(r)
            
        # Check Receipt History
        print("\nReceipt History:")
        query = """
        SELECT TOP 3 l.ITEMNMBR, h.VENDORID, h.RECEIPTDATE, po.DOCDATE
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        JOIN POP30100 po ON l.PONUMBER = po.PONUMBER
        WHERE l.ITEMNMBR = ?
        ORDER BY h.RECEIPTDATE DESC
        """
        cursor.execute(query, item)
        for r in cursor.fetchall():
            print(f"Vendor: {r[1]}, Rcv: {r[2]}, PO: {r[3]}")

    except Exception as e:
        print(f"Error: {e}")
        
    conn.close()

if __name__ == "__main__":
    main()
