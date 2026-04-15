import sys
from db_pool import get_cursor

search_terms = [
    # Manganese
    'Manganese Sulfate', 'Manganese Sucrate', 'Manganese Humate', 'Manganese Chloride', 'Manganese Oxide', 'Manganese EDTA', 'Manganese Chelate', 'Seaweed Extract',
    # Copper
    'Copper Sulfate', 'Copper Sucrate', 'Copper Humate', 'Copper Chloride', 'Copper Oxide', 'Copper EDTA', 'Copper Chelate',
    # Zinc
    'Zinc Sulfate', 'Zinc Sucrate', 'Zinc Humate', 'Zinc Chloride', 'Zinc Oxide', 'Zinc EDTA', 'Zinc Chelate',
    # Iron
    'Iron Sulfate', 'Iron Sucrate', 'Iron Humate', 'Iron Oxide', 'Iron EDTA', 'Iron Chelate',
    # Others
    'Aluminum', 'Elemental Sulfur', 'Sulfur SCU', 'Sulfate Sulfur', 'Boron', 'Molybdenum', 'Cobalt', 'Dolomite', 'Limestone', 'Gypsum', 'Calcium Sulfate',
    # Slow Release
    'Sulfur Coated Urea', 'Resin Coated Urea', 'Methylene Urea', 'IBDU', 'Plastic Coated Urea', 'Polymer Coated Urea'
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
