import sys
from db_pool import get_cursor

query = '''
    SELECT 
        C.PPN_I AS ParentItem, 
        M.ITEMDESC AS ParentDesc,
        C.CPN_I AS ComponentItem,
        CM.ITEMDESC AS ComponentDesc,
        CM.CURRCOST AS CompCost,
        C.QUANTITY_I AS Qty
    FROM BM010115 C
    JOIN IV00101 M ON C.PPN_I = M.ITEMNMBR
    JOIN IV00101 CM ON C.CPN_I = CM.ITEMNMBR
    WHERE C.PPN_I IN ('GOLDMN00', 'GOLDZN00', 'GOLDFE00', 'GOLDCU00', 'GOLDMG00', 'GOLDFEHUM00', 'GOLDCB00')
'''

try:
    with get_cursor() as cursor:
        cursor.execute(query)
        boms = {}
        for row in cursor.fetchall():
            parent = row.ParentItem.strip()
            if parent not in boms:
                boms[parent] = {'desc': row.ParentDesc.strip(), 'comps': []}
            boms[parent]['comps'].append({
                'item': row.ComponentItem.strip(),
                'desc': row.ComponentDesc.strip(),
                'qty': float(row.Qty),
                'cost': float(row.CompCost)
            })

    print("--- 00 ITEMS BOM DERIVATIVE COSTS ---")
    for parent, data in boms.items():
        comps = data['comps']
        total_qty = sum(c['qty'] for c in comps)
        total_cost = sum(c['qty'] * c['cost'] for c in comps)
        if total_qty > 0:
            cost_per_lb = total_cost / total_qty
            cost_per_ton = cost_per_lb * 2000
            print(f"\n{parent}: {data['desc']}")
            print(f"  Total Qty: {total_qty:.4f} lbs | Total Cost: ${total_cost:.4f}")
            print(f"  Calculated Blended Cost: ${cost_per_lb:.4f}/lb -> ${cost_per_ton:.2f}/ton")
            for c in comps:
                print(f"  - {c['item']:<15} | {c['desc']:<50} | Qty: {c['qty']:.2f} | Cost/lb: ${c['cost']:.4f}")

except Exception as e:
    print(f"Error executing query: {e}")
