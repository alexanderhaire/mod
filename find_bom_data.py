import sys
from db_pool import get_cursor

query = '''
    SELECT 
        H.ITEMNMBR AS ParentItem, 
        M.ITEMDESC AS ParentDesc,
        H.CMPTITNM AS ComponentItem,
        CM.ITEMDESC AS ComponentDesc,
        H.Design_Qty AS Qty
    FROM BM00111 H
    JOIN IV00101 M ON H.ITEMNMBR = M.ITEMNMBR
    JOIN IV00101 CM ON H.CMPTITNM = CM.ITEMNMBR
    WHERE 
          M.ITEMDESC LIKE '%Sucrate%' OR 
          M.ITEMDESC LIKE '%Resin%' OR 
          M.ITEMDESC LIKE '%Methylene%' OR 
          M.ITEMDESC LIKE '%IBDU%' OR 
          M.ITEMDESC LIKE '%Dolomite%' OR 
          M.ITEMDESC LIKE '%Limestone%' OR
          M.ITEMDESC LIKE '%Slow Release%' OR
          M.ITEMDESC LIKE '%Polymer%' OR
          M.ITEMDESC LIKE '%Fertilizer%' OR -- Let's just find ANY BOM related to missing terms
          M.ITEMDESC LIKE '%Coated%'
    ORDER BY H.ITEMNMBR
'''

try:
    with get_cursor() as cursor:
        cursor.execute(query)
        print(f"{'Parent':<15} | {'Parent Desc':<40} | {'Component':<15} | {'Comp Desc':<40} | Qty")
        print("-" * 125)
        for row in cursor.fetchall():
            print(f"{row.ParentItem.strip():<15} | {row.ParentDesc.strip()[:38]:<40} | {row.ComponentItem.strip():<15} | {row.ComponentDesc.strip()[:38]:<40} | {row.Qty}")
except Exception as e:
    print(f"Error executing query: {e}")
