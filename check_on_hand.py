
import pyodbc
from db_pool import get_connection

items = ['Hydrogen Peroxide', 'Phosphorous Acid', 'Citric Acid', 'Potassium Acetate', 'Sulfuric Acid']

print("Connecting to DB...")
with get_connection() as conn:
    cursor = conn.cursor()
    print('--- Inventory Lookup ---')
    for item in items:
        # Fuzzy match description
        row = cursor.execute("""
            SELECT TOP 1 i.ITEMNMBR, i.ITEMDESC, q.QTYONHND 
            FROM IV00101 i 
            JOIN IV00102 q ON i.ITEMNMBR = q.ITEMNMBR AND q.LOCNCODE = 'MAIN'
            WHERE i.ITEMDESC LIKE ?
        """, f'%{item}%').fetchone()
        
        if row:
            print(f'{item}: {float(row.QTYONHND):,.0f} (Item: {row.ITEMNMBR})')
        else:
            print(f'{item}: Not Found')
