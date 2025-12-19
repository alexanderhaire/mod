import pyodbc
from secrets_loader import build_connection_string
import datetime

def inspect_data():
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        target_date = '2019-04-03'
        
        # Find which item had a receipt on this date
        query = """
        SELECT 
            l.ITEMNMBR,
            l.POPRCTNM,
            l.UNITCOST,
            l.EXTDCOST,
            l.UMQTYINB,
            l.UOFM
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        WHERE h.RECEIPTDATE = ?
          AND l.ITEMNMBR IN (
              SELECT ITEMNMBR FROM IV00101 WHERE ITEMDESC LIKE '%Omega%' OR ITEMDESC LIKE '%Protein%'
          )
        """
        
        cursor.execute(query, target_date)
        rows = cursor.fetchall()
        
        print(f"--- Receipts on {target_date} ---")
        for row in rows:
            print(f"Item: {row.ITEMNMBR}, Receipt: {row.POPRCTNM}, UnitCost: {row.UNITCOST}, ExtCost: {row.EXTDCOST}, QtyInBase: {row.UMQTYINB}, UofM: {row.UOFM}")
            
            # Check Inventory Layer for this receipt
            inv_query = "SELECT UNITCOST, ADJUNITCOST FROM IV10200 WHERE RCPTNMBR = ? AND ITEMNMBR = ?"
            cursor.execute(inv_query, row.POPRCTNM, row.ITEMNMBR)
            inv_rows = cursor.fetchall()
            for ir in inv_rows:
                print(f"  -> IV10200 Layer Cost: {ir.UNITCOST}, AdjCost: {ir.ADJUNITCOST}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_data()
