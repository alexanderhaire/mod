import pandas as pd
from db_pool import get_cursor

def get_product_prices(items):
    query = f"""
        SELECT ITEMNMBR, ITEMDESC, CURRCOST
        FROM IV00101
        WHERE ITEMNMBR IN ({','.join(['?']*len(set(items)))})
    """
    with get_cursor() as cursor:
        cursor.execute(query, list(set(items)))
        return {row.ITEMNMBR.strip(): {'desc': row.ITEMDESC.strip(), 'cost': float(row.CURRCOST)} for row in cursor.fetchall()}

def get_bom_details(parents):
    query = f'''
        SELECT 
            C.PPN_I AS ParentItem, 
            C.CPN_I AS ComponentItem,
            CM.ITEMDESC AS ComponentDesc,
            CM.CURRCOST AS CompCost,
            C.QUANTITY_I AS Qty
        FROM BM010115 C
        JOIN IV00101 CM ON C.CPN_I = CM.ITEMNMBR
        WHERE C.PPN_I IN ({','.join(['?']*len(set(parents)))})
    '''
    boms = {}
    with get_cursor() as cursor:
        cursor.execute(query, list(set(parents)))
        for row in cursor.fetchall():
            parent = row.ParentItem.strip()
            if parent not in boms:
                boms[parent] = []
            boms[parent].append({
                'item': row.ComponentItem.strip(),
                'desc': row.ComponentDesc.strip(),
                'qty': float(row.Qty),
                'cost': float(row.CompCost)
            })
    return boms

Mappings = [
    # Primary
    ('Total Nitrogen (N)', 'NPKUREA', 46, 'Direct Item'),
    ('Ammoniacal Nitrogen', 'NPKAN', 21, 'Direct Item'),
    ('Nitrate Nitrogen', 'NO3CA', 9, 'Direct Item'),
    ('Water Soluble N / Urea', 'NPKUREA', 46, 'Direct Item'),
    ('Available Phosphorus (P2O5)', 'NPKPHOS75', 54, 'Direct Item'),
    ('Potassium (from Muriate)', 'NPKKCL62', 62, 'Direct Item'), 
    ('Potassium (from Other)', 'NPKKNO3', 45, 'Direct Item'),
    
    # Secondary and Micros
    ('Magnesium', 'NO3MG63', 6.3, 'Direct Item'),
    ('Manganese (from sulfate)', 'SO4MN32', 32, 'Direct Item'),
    ('Manganese (chloride)', 'CL2MN', 16.8, 'Direct Item'),
    ('Manganese (from sucrate)', 'GOLDMN00', 5, 'BOM Derivative'), # derived from Dyna Gold Manganese
    ('Manganese (from oxide)', 'OXMN', 60, 'Direct Item'),
    ('Manganese (from chelate in group 1)', 'EDTAMN', 6, 'Direct Item'),
    ('Copper (from sulfate)', 'SO4CU', 25.2, 'Direct Item'),
    ('Copper (from sucrate)', 'GOLDCU00', 5, 'BOM Derivative'), # derived from Dyna Gold Copper
    ('Copper (from chelate in group 1)', 'EDTACU', 6, 'Direct Item'),
    ('Zinc (from chloride)', 'CL2ZNLIQ', 30.6, 'Direct Item'),
    ('Zinc (from sucrate)', 'GOLDZN00', 7, 'BOM Derivative'), # derived from Dyna Gold Zinc
    ('Zinc (from oxide)', 'OXZN', 80, 'Direct Item'),
    ('Zinc (from chelate in group 1)', 'EDTAZN', 9, 'Direct Item'),
    ('Iron (from sulfate)', 'SO4FEDRY19', 19, 'Direct Item'),
    ('Iron (from sucrate)', 'GOLDFE00', 5, 'BOM Derivative'), # derived from Dyna Gold Iron
    ('Iron Humate', 'GOLDFEHUM00', 5.3, 'BOM Derivative'), # derived from Dyna Gold Iron Humate
    ('Iron (from chelate in group 1)', 'EDTAFEHEDRY', 6, 'Direct Item'),
    ('Sulfur (combined - from Sulfates)', 'CHEH2SO4', 30.3, 'Direct Item'),
    ('Boron', 'SO4BORON', 10, 'Direct Item'),
    ('Cobalt', 'NO3CODRY', 19.8, 'Direct Item'),
    ('Molybdenum', 'SO4MOLY', 39.6, 'Direct Item'),
    ('Calcium (from any source)', 'NO3CA', 11, 'Direct Item'),
    ('Calcium Sulfate (Land Plaster, Gypsum)', 'THIOCACDI', 6, 'Direct Item'),
    ('Seaweed Extract', 'GRPSEAEX', 100, 'Direct Item'),
    ('Urea Triazone', 'NPK3000', 30, 'Direct Item'),
]

# Get all prices
all_items = [m[1] for m in Mappings if m[3] == 'Direct Item']
bom_parents = [m[1] for m in Mappings if m[3] == 'BOM Derivative']
db_prices = get_product_prices(all_items)
bom_details = get_bom_details(bom_parents)

rows = []
for m in Mappings:
    nutrient, item_code, pct, method = m
    
    if method == 'Direct Item':
        if item_code in db_prices:
            p = db_prices[item_code]
            cost_lb = p['cost']
            cost_ton = cost_lb * 2000
            val = (cost_ton) / pct
            
            # Make the math literally readable for the user:
            math_proof = (
                f"1. Item {item_code} costs ${cost_lb:.4f} per lb.\n"
                f"2. 1 Ton (2,000 lbs) costs: ${cost_lb:.4f} * 2000 = ${cost_ton:,.2f} per Ton.\n"
                f"3. Survey Formula: ($ / Ton) / (%)\n"
                f"4. Result: ${cost_ton:,.2f} / {pct} = ${val:,.2f} Per Unit"
            )
            
            rows.append({
                'Survey Category': nutrient,
                'Found GP Item': f"{item_code} - {p['desc']}",
                'Claimed %': f"{pct}%",
                'Cost Per Ton': round(cost_ton, 2),
                'Final Survey Value': round(val, 2),
                'Step-By-Step Math Proof': math_proof,
                'Raw Material BOM Breakdown': "N/A (Standard Raw Material/Finished Good)"
            })
    else:
        if item_code in bom_details:
            comps = bom_details[item_code]
            total_qty = sum(c['qty'] for c in comps)
            total_cost = sum(c['qty'] * c['cost'] for c in comps)
            if total_qty > 0:
                cost_lb = total_cost / total_qty
                cost_ton = cost_lb * 2000
                val = (cost_ton) / pct
                
                # Format component breakdown
                comp_lines = []
                for c in comps:
                    line_cost = c['qty'] * c['cost']
                    comp_lines.append(f"- {c['qty']:.2f} lbs of {c['desc']} (@ ${c['cost']:.4f}/lb) = ${line_cost:.4f}")
                comp_str = "\n".join(comp_lines)
                
                # Step by step math for BOMs
                math_proof = (
                    f"1. A single batch of {item_code} weighs {total_qty:.4f} lbs.\n"
                    f"2. The total sum of all raw material components in the batch is ${total_cost:.4f}.\n"
                    f"3. Blended Cost Per Lb: ${total_cost:.4f} / {total_qty:.4f} lbs = ${cost_lb:.4f}/lb.\n"
                    f"4. 1 Ton (2,000 lbs) formulation cost: ${cost_lb:.4f} * 2000 = ${cost_ton:,.2f} per Ton.\n"
                    f"5. Survey Formula: ($ / Ton) / (%)\n"
                    f"6. Result: ${cost_ton:,.2f} / {pct} = ${val:,.2f} Per Unit"
                )

                rows.append({
                    'Survey Category': nutrient,
                    'Found GP Item': f"{item_code} - Derived From Raw Material Mixture",
                    'Claimed %': f"{pct}%",
                    'Cost Per Ton': round(cost_ton, 2),
                    'Final Survey Value': round(val, 2),
                    'Step-By-Step Math Proof': math_proof,
                    'Raw Material BOM Breakdown': comp_str
                })

df = pd.DataFrame(rows)

out_file = r'c:\Users\alexh\Downloads\mod\Commercial_Values_Math_Proof.xlsx'

with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Survey Calculations', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['Survey Calculations']
    
    # Formats
    header_format = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1
    })
    money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
    wrap_format = workbook.add_format({'valign': 'top', 'text_wrap': True})
    
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        
    worksheet.set_column('A:A', 35, wrap_format)    # Category
    worksheet.set_column('B:B', 35, wrap_format)    # Matched Item
    worksheet.set_column('C:C', 10, wrap_format)    # Pct
    worksheet.set_column('D:D', 15, money_fmt)      # Cost Ton
    worksheet.set_column('E:E', 18, money_fmt)      # Final Value
    worksheet.set_column('F:F', 70, wrap_format)    # Math Proof
    worksheet.set_column('G:G', 80, wrap_format)    # BOM

print(f"Proof saved to {out_file}")
