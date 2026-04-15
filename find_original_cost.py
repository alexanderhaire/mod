import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def trace_po_costs(po_number, item_number):
    print(f"--- COST TRACE: PO {po_number} | Item {item_number} ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Check PO Line Work/History for the BASE UNIT COST
        # POP10110 is Work, POP30110 is History
        print("\n1. Checking PO Line Tables (Base Unit Cost)...")
        query_po_lines = f"""
        SELECT 'WORK' as Source, PONUMBER, ITEMNMBR, UNITCOST, EXTDCOST 
        FROM POP10110 WHERE PONUMBER = '{po_number}' AND ITEMNMBR = '{item_number}'
        UNION ALL
        SELECT 'HIST' as Source, PONUMBER, ITEMNMBR, UNITCOST, EXTDCOST 
        FROM POP30110 WHERE PONUMBER = '{po_number}' AND ITEMNMBR = '{item_number}'
        """
        po_data = pd.read_sql(query_po_lines, conn)
        print(po_data.to_string() if not po_data.empty else "No PO line data found.")

        # 2. Check Receipt Line items for the RECEIVED COST
        print("\n2. Checking Receipt Line History (POP30310)...")
        try:
            query_rct_hist = f"""
            SELECT 'HIST' as Source, POPRCTNM, PONUMBER, ITEMNMBR, UNITCOST 
            FROM POP30310 WHERE PONUMBER = '{po_number}' AND ITEMNMBR = '{item_number}'
            """
            receipt_data = pd.read_sql(query_rct_hist, conn)
            print(receipt_data.to_string() if not receipt_data.empty else "No receipt history found.")
        except Exception as e:
            print(f"POP30310 Error: {e}")
            receipt_data = pd.DataFrame()

        # 4. Check Inventory Layers (IV10200)
        print("\n4. Checking Inventory Layers (IV10200 - Purchase Receipts)...")
        try:
            # Note: RCTSEQNM is the sequence number (not RCPTSEQNM or RCPTSQTN)
            query_iv = f"""
            SELECT ITEMNMBR, RCPTNMBR, RCTSEQNM, UNITCOST, ADJUNITCOST, QTYRECVD, QTYSOLD 
            FROM IV10200 
            WHERE ITEMNMBR = '{item_number}'
            ORDER BY DATERECD DESC
            """
            iv_data = pd.read_sql(query_iv, conn)
            print(iv_data.head(10).to_string() if not iv_data.empty else "No inventory layer data found.")
        except Exception as e:
            print(f"IV10200 Error: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    # From screenshot: PO 431-8232, Item NO3CA
    trace_po_costs("431-8232", "NO3CA")
