import pyodbc
from secrets_loader import build_connection_string

def get_connection():
    conn_str, _, _, _ = build_connection_string()
    return pyodbc.connect(conn_str)

def debug_inventory(item_nmbr):
    conn = get_connection()
    cursor = conn.cursor()
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING ITEM: {item_nmbr}")
    print('='*60)
    
    # 1. Check IV00102 (Inventory Quantities)
    print("\n[IV00102 - Inventory Quantities]")
    cursor.execute("SELECT LOCNCODE, RCRDTYPE, QTYONHND, ATYALLOC, QTYONORD FROM IV00102 WHERE ITEMNMBR = ?", item_nmbr)
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    print(f"Columns: {columns}")
    total_iv_ord = 0
    for row in rows:
        print(f"  {row}")
        total_iv_ord += row.QTYONORD
    print(f">>> Total IV00102.QTYONORD: {total_iv_ord}")

    # 2. Check POP10110 (PO Lines)
    print("\n[POP10110 - Purchase Order Lines]")
    cursor.execute("""
        SELECT 
            t1.PONUMBER, 
            t1.ITEMNMBR, 
            t1.QTYORDER, 
            t1.QTYCANCE, 
            t1.QTYSHPPD, 
            t1.QTYINVCD, 
            t1.POLNESTA,
            t2.POSTATUS
        FROM POP10110 t1
        JOIN POP10100 t2 ON t1.PONUMBER = t2.PONUMBER
        WHERE t1.ITEMNMBR = ?
    """, item_nmbr)
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    print(f"Columns: {columns}")
    
    calc_on_order_invcd = 0
    calc_on_order_shppd = 0
    
    for row in rows:
        print(f"  {row}")
        rem_inv = row.QTYORDER - row.QTYCANCE - row.QTYINVCD
        rem_shp = row.QTYORDER - row.QTYCANCE - row.QTYSHPPD
        
        # Only count if positive
        rem_inv = max(0, rem_inv)
        rem_shp = max(0, rem_shp)
        
        calc_on_order_invcd += rem_inv
        calc_on_order_shppd += rem_shp
        
    print(f"\n>>> Calculated OnOrder (QTYORDER - QTYCANCE - QTYINVCD): {calc_on_order_invcd}")
    print(f">>> Calculated OnOrder (QTYORDER - QTYCANCE - QTYSHPPD): {calc_on_order_shppd}")
    
    conn.close()

if __name__ == "__main__":
    debug_inventory('NO3CA')
    print("\n" + "="*80 + "\n")
    debug_inventory('NO3CA12')
