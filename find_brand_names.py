import sys
from db_pool import get_cursor

search_terms = [
    'Polyon', 'Osmocote', 'Nutricote', 'Duration', 'ESN', 'Agrocote', # Coated ureas
    'Nitroform', 'Nutralene', 'Ureaform', 'UF ', ' UF', # Methylene ureas
    'Ag Lime', 'Aglime', 'Hi-Cal', # Limestones 
    'Isobutyl', # IBDU
    'SCU', # Sulfur coated
]

conditions = []
for term in search_terms:
    conditions.append(f"ITEMDESC LIKE '%{term}%'")

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
