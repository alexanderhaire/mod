import pyodbc
from secrets_loader import build_connection_string

def debug():
    conn_str, _, _, _ = build_connection_string()
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        item = 'SOARCIT00'
        
        print(f"Checking max dates for {item}...")
        
        # SOP30200 (Sales Header) linked to SOP30300
        query_sop = """
        SELECT MAX(h.DOCDATE) 
        FROM SOP30300 t
        JOIN SOP30200 h ON t.SOPNUMBE = h.SOPNUMBE AND t.SOPTYPE = h.SOPTYPE
        WHERE t.ITEMNMBR = ?
        """
        cursor.execute(query_sop, item)
        print(f"Max Sales Date: {cursor.fetchone()[0]}")
        
        # POP30300 (Purchase Header) linked to POP30310
        query_pop = """
        SELECT MAX(h.RECEIPTDATE)
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE l.ITEMNMBR = ?
        """
        cursor.execute(query_pop, item)
        print(f"Max Purchase Date: {cursor.fetchone()[0]}")

        # IV30200 (Inventory Header) linked to IV30300
        query_iv = """
        SELECT MAX(h.DOCDATE)
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        WHERE t.ITEMNMBR = ?
        """
        cursor.execute(query_iv, item)
        print(f"Max Inventory Date: {cursor.fetchone()[0]}")

if __name__ == "__main__":
    debug()
