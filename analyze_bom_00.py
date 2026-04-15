import sys
from db_pool import get_cursor
from collections import defaultdict

# We want to find 00 items whose BOM contains specific marker ingredients
markers = {
    'Sugar/Molasses (Sucrate)': ['MOLASSES', 'SUGAR', 'SUCROSE', 'CHEMOLASS', 'FLOMOLA', 'ACS030'],
    'Humic/Humate': ['HUMIC', 'HUMATE', 'FLOHUM', 'GOLDHUM'],
    'Chloride': ['CHLORIDE', 'CL2'],
    'Oxide': ['OXIDE', 'OXZN', 'OXMN', 'OXMG'],
    'Urea/Slow': ['UREA', 'TRIAZONE', 'METHYLENE', 'COAT', 'NPK3000', 'NPKUREA']
}

metals = {
    'Manganese': ['MN', 'MANGANESE', 'SO4MN'],
    'Zinc': ['ZN', 'ZINC', 'SO4ZN'],
    'Copper': ['CU', 'COPPER', 'SO4CU'],
    'Iron': ['FE', 'IRON', 'SO4FE'],
    'Calcium': ['CA', 'CALCIUM', 'NO3CA', 'THIOCA'],
    'Magnesium': ['MG', 'MAGNESIUM', 'SO4MG']
}

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
    WHERE H.ITEMNMBR LIKE '%00'
'''

boms = defaultdict(list)

try:
    with get_cursor() as cursor:
        cursor.execute(query)
        for row in cursor.fetchall():
            parent = row.ParentItem.strip()
            parent_desc = row.ParentDesc.strip()
            comp = row.ComponentItem.strip()
            comp_desc = row.ComponentDesc.strip()
            boms[(parent, parent_desc)].append((comp, comp_desc, row.Qty))
            
    print("--- POTENTIAL SUCRATES (Metal + Molasses/Sugar) ---")
    for (p, p_desc), comps in boms.items():
        has_sugar = any(any(m in c[0] or m in c[1].upper() for m in markers['Sugar/Molasses (Sucrate)']) for c in comps)
        has_metal = any(any(m in c[0] or m in c[1].upper() for metal_list in metals.values() for m in metal_list) for c in comps)
        
        if has_sugar and has_metal:
            print(f"\nParent: {p} - {p_desc}")
            for c in comps:
                print(f"  - {c[0]:<15} : {c[1]}")

    print("\n--- POTENTIAL HUMATES (Metal + Humic) ---")
    for (p, p_desc), comps in boms.items():
        has_humic = any(any(m in c[0] or m in c[1].upper() for m in markers['Humic/Humate']) for c in comps)
        has_metal = any(any(m in c[0] or m in c[1].upper() for metal_list in metals.values() for m in metal_list) for c in comps)
        
        if has_humic and has_metal:
            print(f"\nParent: {p} - {p_desc}")
            for c in comps:
                print(f"  - {c[0]:<15} : {c[1]}")

    print("\n--- SLOW RELEASE / COATED UREA (Urea + Polymers/Coating/Triazone) ---")
    for (p, p_desc), comps in boms.items():
        # Check if it has Urea AND some other slow release agent. Actually any Urea might be processed.
        has_urea = any(any(m in c[0] or m in c[1].upper() for m in ['UREA', 'NPKUREA']) for c in comps)
        has_slow = any(any(m in c[0] or m in c[1].upper() for m in ['TRIAZONE', 'METH', 'FORMALDEHYDE', 'COAT', 'POLY']) for c in comps)
        
        if has_urea and has_slow:
            print(f"\nParent: {p} - {p_desc}")
            for c in comps:
                print(f"  - {c[0]:<15} : {c[1]}")
                
except Exception as e:
    print(f"Error executing query: {e}")
