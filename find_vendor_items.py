import sys
from db_pool import get_cursor

query = '''
    SELECT ITEMNMBR, VNDITNUM, VNDITDSC
    FROM IV00103
    WHERE VNDITDSC LIKE '%Sucrate%'
       OR VNDITDSC LIKE '%Methylene%'
       OR VNDITDSC LIKE '%IBDU%'
       OR VNDITDSC LIKE '%Resin%'
       OR VNDITDSC LIKE '%Coated%'
       OR VNDITDSC LIKE '%Dolomite%'
       OR VNDITDSC LIKE '%Limestone%'
'''

try:
    with get_cursor() as cursor:
        cursor.execute(query)
        print(f"{'Item':<15} | {'Vendor Item':<20} | Vendor Desc")
        print("-" * 100)
        for row in cursor.fetchall():
            print(f"{row.ITEMNMBR.strip():<15} | {row.VNDITNUM.strip():<20} | {row.VNDITDSC.strip()}")
except Exception as e:
    print(f"Error executing query: {e}")
