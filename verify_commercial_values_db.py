import sys
from db_pool import get_cursor

items = [
    'NPKUREA', 'NPKAN', 'NO3CA', 'NPKPHOS75', 'NPKKNO3', 'NO3MG63', 
    'NO3MN', 'NO3CU', 'NO3ZN', 'NO3FE', 'CHEH2SO4'
]

query = f"""
    SELECT ITEMNMBR, ITEMDESC, CURRCOST, STNDCOST
    FROM IV00101
    WHERE ITEMNMBR IN ({','.join(['?']*len(items))})
"""

print(f"{'Item':<15} | {'Desc':<30} | {'Current Cost':<15} | {'Stnd Cost':<15}")
print("-" * 80)

try:
    with get_cursor() as cursor:
        cursor.execute(query, items)
        for row in cursor.fetchall():
            print(f"{row.ITEMNMBR.strip():<15} | {row.ITEMDESC.strip():<30} | {row.CURRCOST:<15.4f} | {row.STNDCOST:<15.4f}")
except Exception as e:
    print(f"Error querying db: {e}")
