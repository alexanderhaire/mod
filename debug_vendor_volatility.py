
import pyodbc
from secrets_loader import build_connection_string

def main():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    item = 'NPK15012'
    print(f"\n--- Analyzing Vendor Volatility for {item} ---")
    
    # 1. Who is the Primary Vendor?
    cursor.execute("SELECT PRIMVNDR FROM IV00102 WHERE ITEMNMBR = ? AND RCRDTYPE = 1", (item,))
    row = cursor.fetchone()
    primary_vendor = row.PRIMVNDR.strip() if row else "UNKNOWN"
    
    print(f"Primary Vendor (IV00102): {primary_vendor}")

    # Check IV00103 (Item Vendor Mapping)
    cursor.execute("SELECT VENDORID, LSTORDDT FROM IV00103 WHERE ITEMNMBR = ? ORDER BY LSTORDDT DESC", (item,))
    valid_vendors = cursor.fetchall()
    print("Vendors in IV00103 (Vendor Master for Item):")
    for v in valid_vendors:
        print(f"  - {v.VENDORID.strip()} (Last Order: {v.LSTORDDT})")

    # 2. Who are we buying from?
    cursor.execute("""
        SELECT h.VENDORID, COUNT(*) as POs, 
               STDEV(DATEDIFF(day, po.DOCDATE, h.RECEIPTDATE)) as Vol
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        JOIN POP30100 po ON l.PONUMBER = po.PONUMBER
        WHERE l.ITEMNMBR = ?
        GROUP BY h.VENDORID
    """, (item,))
    
    rows = cursor.fetchall()
    for r in rows:
        print(f"  Vendor: {r.VENDORID.strip()} | POs: {r.POs} | Vol: {r.Vol}")

if __name__ == "__main__":
    main()
