import sys
from db_pool import get_cursor

query = '''
    SELECT 
        H.ITEMNMBR AS ParentItem, 
        M.ITEMDESC AS ParentDesc,
        C.CPN_I AS ComponentItem,
        CM.ITEMDESC AS ComponentDesc,
        C.Design_Qty_I AS Qty
    FROM BM010115 C
    JOIN BM00111 H ON C.BM_TRX_ID_I = H.BM_TRX_ID_I
    JOIN IV00101 M ON H.ITEMNMBR = M.ITEMNMBR
    JOIN IV00101 CM ON C.CPN_I = CM.ITEMNMBR
    WHERE H.ITEMNMBR LIKE '%00' 
      AND (
          M.ITEMDESC LIKE '%Sucrate%' OR 
          M.ITEMDESC LIKE '%Resin%' OR 
          M.ITEMDESC LIKE '%Methylene%' OR 
          M.ITEMDESC LIKE '%IBDU%' OR 
          M.ITEMDESC LIKE '%Dolomite%' OR 
          M.ITEMDESC LIKE '%Limestone%' OR
          M.ITEMDESC LIKE '%Coated%'
      )
'''

# Let's just broadly search for any BOMs matching the missing terms 
# regardless of if it ends in 00 just to be safe, but prioritize them.

query_broad = '''
    SELECT 
        C.ITEMNMBR AS ParentItem, 
        M.ITEMDESC AS ParentDesc,
        C.CMPTITNM AS ComponentItem,
        CM.ITEMDESC AS ComponentDesc,
        C.Design_Qty_I AS Qty
    FROM BM010115 C
    JOIN IV00101 M ON C.ITEMNMBR = M.ITEMNMBR
    JOIN IV00101 CM ON C.CMPTITNM = CM.ITEMNMBR
    WHERE 
          M.ITEMDESC LIKE '%Sucrate%' OR 
          M.ITEMDESC LIKE '%Resin%' OR 
          M.ITEMDESC LIKE '%Methylene%' OR 
          M.ITEMDESC LIKE '%IBDU%' OR 
          M.ITEMDESC LIKE '%Dolomite%' OR 
          M.ITEMDESC LIKE '%Limestone%' OR
          M.ITEMDESC LIKE '%Coated%'
'''

# Above tables might be wrong fields, BM010115 has ITEMNMBR and CMPTITNM in typical GP
# Let's just inspect BM010115 columns
query_schema = "SELECT TOP 1 * FROM BM010115"

try:
    with get_cursor() as cursor:
        cursor.execute(query_schema)
        columns = [column[0] for column in cursor.description]
        print(f"BM010115 Columns: {columns}")
        
        cursor.execute("SELECT TOP 1 * FROM BM00111")
        columns = [column[0] for column in cursor.description]
        print(f"BM00111 Columns: {columns}")
except Exception as e:
    print(f"Error executing query: {e}")
