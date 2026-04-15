import sys
from db_pool import get_cursor

query = '''
    SELECT ITEMNMBR, ITEMDESC, CURRCOST
    FROM IV00101
    WHERE ITEMDESC LIKE '%KCL%' 
       OR ITEMDESC LIKE '%Potash%' 
       OR ITEMDESC LIKE '%Muriate%' 
       OR ITEMDESC LIKE '%0-0-60%'
       OR ITEMDESC LIKE '%0-0-62%'
       OR ITEMNMBR LIKE '%KCL%'
'''

with get_cursor() as cursor:
    cursor.execute(query)
    for row in cursor.fetchall():
        print(f'{row.ITEMNMBR.strip():<15} | {row.ITEMDESC.strip():<40} | {row.CURRCOST:<15.4f}')
