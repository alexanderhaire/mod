import sys
from db_pool import get_cursor

def calculate_unit_value(price_per_lb, percentage):
    if percentage == 0: return 0
    price_per_ton = price_per_lb * 2000
    return price_per_ton / percentage

# Using standard known percentages from descriptions / industry norms where omitted
mappings = [
    # Primary
    ('Total Nitrogen (N)', 'NPKUREA', 46),
    ('Ammoniacal Nitrogen', 'NPKAN', 21),
    ('Nitrate Nitrogen', 'NO3CA', 9),
    ('Water Soluble N / Urea', 'NPKUREA', 46),
    ('Available Phosphorus (P2O5)', 'NPKPHOS75', 54),
    ('Potassium (from Muriate)', 'NPKKCL62', 62), 
    ('Potassium (from Other)', 'NPKKNO3', 45),
    
    # Secondary / Micros
    ('Magnesium', 'NO3MG63', 6.3),
    
    ('Manganese (from sulfate)', 'SO4MN32', 32),
    ('Manganese (chloride)', 'CL2MN', 16.8),
    ('Manganese (from chelate in group 1)', 'EDTAMN', 6),
    ('Manganese (from oxide)', 'OXMN', 60), # Manganous oxide is typically 60-62% Mn
    
    ('Copper (from sulfate)', 'SO4CU', 25.2),
    ('Copper (from chelate in group 1)', 'EDTACU', 6),
    
    ('Zinc (from chloride)', 'CL2ZNLIQ', 30.6),
    ('Zinc (from oxide)', 'OXZN', 80), # 99.9% ZnO is 80.3% Zn, will use 80%
    ('Zinc (from chelate in group 1)', 'EDTAZN', 9),
    
    ('Iron (from sulfate)', 'SO4FEDRY19', 19),
    ('Iron Humate', 'GOLDFEHUM00', 5.3), 
    ('Iron (from chelate in group 1)', 'EDTAFEHEDRY', 6), 
    
    ('Sulfur (combined - from Sulfates)', 'CHEH2SO4', 30.3),
    
    ('Boron', 'SO4BORON', 10), 
    ('Cobalt', 'NO3CODRY', 19.8),
    ('Molybdenum', 'SO4MOLY', 39.6), 
    
    ('Calcium (from any source)', 'NO3CA', 11),
    
    # Slow Release
    ('Urea Triazone', 'NPK3000', 30), # Triazone 30-0-0 has 30% N

    # Other
    ('Calcium Sulfate (Land Plaster, Gypsum)', 'THIOCACDI', 6), 
    ('Seaweed Extract', 'GRPSEAEX', 100), 
]

items = [m[1] for m in mappings]
query = f"""
    SELECT ITEMNMBR, ITEMDESC, CURRCOST
    FROM IV00101
    WHERE ITEMNMBR IN ({','.join(['?']*len(set(items)))})
"""

with get_cursor() as cursor:
    cursor.execute(query, list(set(items)))
    db_prices = {row.ITEMNMBR.strip(): {'desc': row.ITEMDESC.strip(), 'cost': float(row.CURRCOST)} for row in cursor.fetchall()}

with open('commercial_values_report.md', 'w') as f:
    f.write('# Fertilizer Commercial Values Survey 25-26 Report\n\n')
    f.write('Based on live real-time data from the Great Plains database, here are the calculated **Commercial Values (per unit)** for the survey.\n\n')
    f.write('**Note:** A "Unit" is defined as 20 lbs (1% of a ton).\n')
    f.write('**Calculation Method:** `(Price per lb * 2000) / Nutrient Percentage`\n\n')

    f.write('## 1. Primary Plant Nutrients\n\n')
    f.write('| Nutrient | Source Found | Price/Ton | % | **Commercial Value ($/Unit)** |\n')
    f.write('| :--- | :--- | :--- | :--- | :--- |\n')

    current_section = 1
    for m in mappings:
        if m[0] == 'Magnesium' and current_section == 1:
            f.write('\n## 2. Secondary Plant Nutrients\n\n')
            f.write('| Nutrient | Source Found | Price/Ton | % | **Commercial Value ($/Unit)** |\n')
            f.write('| :--- | :--- | :--- | :--- | :--- |\n')
            current_section = 2
            
        if m[0] == 'Urea Triazone':
            f.write('\n## Slow Release Materials\n\n')
            f.write('| Nutrient | Source Found | Price/Ton | % | **Commercial Value ($/Unit)** |\n')
            f.write('| :--- | :--- | :--- | :--- | :--- |\n')

        if m[0] == 'Calcium Sulfate (Land Plaster, Gypsum)':
            f.write('\n## Other Categories\n\n')
            f.write('| Nutrient | Source Found | Price/Ton | % | **Commercial Value ($/Unit)** |\n')
            f.write('| :--- | :--- | :--- | :--- | :--- |\n')
            
        item_code = m[1]

        if item_code in db_prices:
            p = db_prices[item_code]
            price_ton = p['cost'] * 2000
            val = calculate_unit_value(p['cost'], m[2])
            f.write(f'| **{m[0]}** | {p["desc"]} | ${price_ton:.2f} | {m[2]}% | **${val:.2f}** |\n')
        else:
             f.write(f'| **{m[0]}** | *Not Found* | - | - | *unknown* |\n')

    f.write('\n## Missing Information\n')
    f.write('* Many specific compound types (sucrates, specific slow release resins) had no matching exact inventory item with cost data.\n')
    f.write('* *Calcium assumes 19% Ca for Calcium Nitrate 11% (CN9)*\n')
