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
    WHERE C.PPN_I LIKE '%00'
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

    print(f"Loaded {len(boms)} parent items with BOMs.")
    
    sugar_terms = ['MOLASSES', 'CHEMOLASS', 'SUGAR', 'SUCROSE']
    metal_terms = ['MANGANESE', 'SO4MN', 'ZINC', 'SO4ZN', 'COPPER', 'SO4CU', 'IRON', 'SO4FE']
    
    print("--- 00 ITEMS WITH METAL AND SUGAR (SUCRATE DERIVATIVES) ---")
    for parent, data in boms.items():
        comps = data['comps']
        comp_descs = [c['desc'].upper() for c in comps]
        
        has_sugar = any(any(s in d for s in sugar_terms) for d in comp_descs)
        has_metal = any(any(m in d for m in metal_terms) for d in comp_descs)
        
        if has_sugar and has_metal:
            print(f"\nParent: {parent} - {data['desc']}")
            total_qty = sum(c['qty'] for c in comps)
            total_cost = sum(c['qty'] * c['cost'] for c in comps)
            if total_qty > 0:
                cost_per_lb = total_cost / total_qty
                print(f"  Calculated Blended Cost: ${cost_per_lb:.4f}/lb")
                for c in comps:
                    print(f"  - {c['item']:<15} | {c['desc']:<50} | Qty: {c['qty']:.2f} | Cost/lb: ${c['cost']:.4f}")

    # Also search for humates
    humic_terms = ['HUMIC', 'HUMATE']
    print("\n--- 00 ITEMS WITH METAL AND HUMIC (HUMATE DERIVATIVES) ---")
    for parent, data in boms.items():
        comps = data['comps']
        comp_descs = [c['desc'].upper() for c in comps]
        
        has_humic = any(any(h in d for h in humic_terms) for d in comp_descs)
        has_metal = any(any(m in d for m in metal_terms) for d in comp_descs)
        
        if has_humic and has_metal:
            print(f"\nParent: {parent} - {data['desc']}")
            total_qty = sum(c['qty'] for c in comps)
            total_cost = sum(c['qty'] * c['cost'] for c in comps)
            if total_qty > 0:
                cost_per_lb = total_cost / total_qty
                print(f"  Calculated Blended Cost: ${cost_per_lb:.4f}/lb")
                for c in comps:
                    print(f"  - {c['item']:<15} | {c['desc']:<50} | Qty: {c['qty']:.2f} | Cost/lb: ${c['cost']:.4f}")

    # Also search for slow release urea
    urea_terms = ['UREA', 'NPKUREA']
    slow_terms = ['TRIAZONE', 'METH', 'FORMALDEHYDE', 'COAT', 'POLY']
    print("\n--- SLOW RELEASE / COATED UREA DERIVATIVES ---")
    for parent, data in boms.items():
        comps = data['comps']
        comp_descs = [c['desc'].upper() for c in comps]
        
        has_urea = any(any(u in d for u in urea_terms) for d in comp_descs)
        has_slow = any(any(s in d for s in slow_terms) for d in comp_descs)
        
        if has_urea and has_slow:
            print(f"\nParent: {parent} - {data['desc']}")
            total_qty = sum(c['qty'] for c in comps)
            total_cost = sum(c['qty'] * c['cost'] for c in comps)
            if total_qty > 0:
                cost_per_lb = total_cost / total_qty
                print(f"  Calculated Blended Cost: ${cost_per_lb:.4f}/lb")
                for c in comps:
                    print(f"  - {c['item']:<15} | {c['desc']:<50} | Qty: {c['qty']:.2f} | Cost/lb: ${c['cost']:.4f}")


except Exception as e:
    print(f"Error executing query: {e}")
