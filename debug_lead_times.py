
import pyodbc
from secrets_loader import build_connection_string

def test_lead_time_calc():
    conn_str, server, db, auth = build_connection_string()
    try:
        with pyodbc.connect(conn_str, autocommit=True) as conn:
            cursor = conn.cursor()
            
            # Try to link Receipt History (POP30300/310) to PO History (POP30100)
            # We want to find: 
            # 1. A receipt line (POP30310) that has a PONUMBER
            # 2. Link to Receipt Header (POP30300) for Receipt Date
            # 3. Link to PO History Header (POP30100) for PO Date
            
            query = """
            SELECT TOP 20
                r_line.ITEMNMBR,
                r_head.RECEIPTDATE,
                po_head.DOCDATE as PO_Date,
                po_head.PONUMBER,
                r_head.VENDNAME,
                DATEDIFF(day, po_head.DOCDATE, r_head.RECEIPTDATE) as LeadTimeDays
            FROM POP30310 r_line
            JOIN POP30300 r_head ON r_line.POPRCTNM = r_head.POPRCTNM
            JOIN POP30100 po_head ON r_line.PONUMBER = po_head.PONUMBER
            WHERE r_line.PONUMBER <> '' 
              AND po_head.DOCDATE IS NOT NULL
              AND r_head.RECEIPTDATE IS NOT NULL
            ORDER BY r_head.RECEIPTDATE DESC
            """
            
            print("--- Testing Historical Lead Time Calculation (Closed POs) ---")
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                print("No rows found linking Receipts to PO History.")
            else:
                for row in rows:
                    print(f"Item: {row.ITEMNMBR} | PO: {row.PONUMBER} | Order: {row.PO_Date} -> Recv: {row.RECEIPTDATE} | Lead Time: {row.LeadTimeDays} days")

            # Also check Open POs (POP10100) just in case some receipts are partial and PO is still open
            query_open = """
            SELECT TOP 5
                r_line.ITEMNMBR,
                r_head.RECEIPTDATE,
                po_head.DOCDATE as PO_Date,
                po_head.PONUMBER,
                DATEDIFF(day, po_head.DOCDATE, r_head.RECEIPTDATE) as LeadTimeDays
            FROM POP30310 r_line
            JOIN POP30300 r_head ON r_line.POPRCTNM = r_head.POPRCTNM
            JOIN POP10100 po_head ON r_line.PONUMBER = po_head.PONUMBER
            WHERE r_line.PONUMBER <> ''
            ORDER BY r_head.RECEIPTDATE DESC
            """
            print("\n--- Testing Open PO Links (Partial Receipts) ---")
            cursor.execute(query_open)
            rows_open = cursor.fetchall()
            if not rows_open:
                print("No rows found linking Receipts to Open POs.")
            else:
                for row in rows_open:
                    print(f"Item: {row.ITEMNMBR} | PO: {row.PONUMBER} | Order: {row.PO_Date} -> Recv: {row.RECEIPTDATE} | Lead Time: {row.LeadTimeDays} days")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_lead_time_calc()
