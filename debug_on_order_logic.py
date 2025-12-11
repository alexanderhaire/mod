import pyodbc
from secrets_loader import build_connection_string

def find_on_order_logic():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'NO3CA12'
        
        # Get all PO lines for the item
        query = """
        SELECT 
            l.PONUMBER, 
            l.QTYORDER, 
            l.QTYCANCE, 
            l.POLNESTA, 
            h.POSTATUS,
            l.LOCNCODE
        FROM POP10110 l
        JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
        WHERE l.ITEMNMBR = ?
        """
        cursor.execute(query, item)
        rows = cursor.fetchall()
        
        print(f"Total PO Lines found: {len(rows)}")
        
        # Filter for MAIN location as requested
        main_rows = [r for r in rows if r.LOCNCODE.strip() == 'MAIN']
        print(f"Rows in MAIN: {len(main_rows)}")
        
        # Try different sums
        total_open = 0
        details = []
        
        # PO Status: 1=New, 2=Submitted, 3=Partially Received, 4=Closed, 5=Canceled, 6=Voided
        # Line Status details might vary, but usually < 4 or 5 is open.
        
        for r in main_rows:
            qty = r.QTYORDER - r.QTYCANCE
            # Basic Open Check: PO Status < 4 (New, Submitted, Partial) and Line Status < 5 (Closed)
            # Adjust these numbers based on findings
            
            # Let's just sum everything that looks "Open"
            if r.POSTATUS in (1, 2, 3) and r.POLNESTA not in (5, 6):
                 total_open += qty
                 details.append((r.PONUMBER, qty, r.POSTATUS, r.POLNESTA))

        print(f"Calculated Total On Order (Status < 4, Line < 5, MAIN): {total_open}")
        
        if total_open == 132000:
            print("MATCH FOUND!")
        else:
            print("No Exact Match yet. Listing components:")
            for d in details:
                print(d)
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    find_on_order_logic()
