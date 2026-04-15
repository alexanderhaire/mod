import sys
from db_pool import get_cursor

search_terms = [
    'molasses', 'sugar', 'sucrose', # For sucrates
    'dolomite', 'limestone', 'calcium carbonate', 'magnesium carbonate',
    'resin', 'polymer', 'coating', 'sulfur coat',
    'formaldehyde', 'methylene', 'ibdu', 'triazone'
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
    WHERE {' OR '.join(conditions)}
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
