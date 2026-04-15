import sys
from db_pool import get_cursor

search_terms = [
    'Sucrate', 'Manganese Sucrate', 'Copper Sucrate', 'Zinc Sucrate', 'Iron Sucrate',
    'Oxide', 'Manganese Oxide', 'Copper Oxide', 'Iron Oxide',
    'Sulfur Coated', 'Resin Coated', 'Methylene', 'IBDU', 'Plastic Coated', 'Polymer Coated',
    'Dolomite', 'Limestone',
    'Triazone'
]

conditions = []
for term in search_terms:
    words = term.split()
    word_conds = [f"ITEMDESC LIKE '%{w}%'" for w in words]
    combined_cond = " AND ".join(word_conds)
    conditions.append(f"({combined_cond})")

query = f'''
    SELECT ITEMNMBR, ITEMDESC, CURRCOST
    FROM IV00101
    WHERE ITEMTYPE = 1 AND ({' OR '.join(conditions)}) -- ITEMTYPE 1 usually denotes finished goods or we can just search all
'''
# Let's search all first to avoid missing things due to ITEMTYPE, but prioritize items that might be finished goods (often different ITEMNMBR prefixes)

query = f'''
    SELECT ITEMNMBR, ITEMDESC, CURRCOST
    FROM IV00101
    WHERE ({' OR '.join(conditions)})
    AND ITEMNMBR NOT LIKE 'CHEMOLASS%'
'''

try:
    with get_cursor() as cursor:
        cursor.execute(query)
        print(f"{'Item':<15} | {'Desc':<60} | Current Cost")
        print("-" * 100)
        for row in cursor.fetchall():
            print(f"{row.ITEMNMBR.strip():<15} | {row.ITEMDESC.strip():<60} | {row.CURRCOST:<15.4f}")
except Exception as e:
    print(f"Error executing query: {e}")
