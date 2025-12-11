import pyodbc
from secrets_loader import build_connection_string

def analyze_po_statuses():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item = 'NO3CA12'
        
        # Get all PO lines for the item with statuses
        # Note: POP10100 POSTATUS: 1=New, 2=Released, 3=Change Order, 4=Received, 5=Closed, 6=Canceled
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
        WHERE l.ITEMNMBR = ? AND l.LOCNCODE = 'MAIN'
        """
        cursor.execute(query, item)
        rows = cursor.fetchall()
        
        print(f"{'PONUMBER':<15} {'QTY':<10} {'PO_STS':<7} {'LINE_STS':<8}")
        print("-" * 45)
        
        total_qty = 0
        for r in rows:
            qty = r.QTYORDER - r.QTYCANCE
            print(f"{r.PONUMBER:<15} {qty:<10.2f} {r.POSTATUS:<7} {r.POLNESTA:<8}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    analyze_po_statuses()
