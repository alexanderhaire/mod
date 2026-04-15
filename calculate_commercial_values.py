import csv

def load_prices(filepath):
    prices = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = row['ITEMNMBR'].strip()
            desc = row['ITEMDESC'].strip()
            try:
                cost = float(row['CURRCOST'])
                prices[item] = {'desc': desc, 'cost_lb': cost}
            except ValueError:
                continue
    return prices

def calculate_unit_value(price_per_lb, percentage):
    if percentage == 0: return 0
    price_per_ton = price_per_lb * 2000
    # Unit is 1% of a ton (20 lbs)
    # Value per Unit = Price per Ton / Percentage
    return price_per_ton / percentage

def main():
    prices = load_prices(r'c:\Users\alexh\Downloads\mod\price_to_beat.csv')
    
    print("=== FERTILIZER COMMERCIAL VALUES CALCULATION ===")
    print(f"{'Category':<30} | {'Item':<15} | {'Desc':<30} | {'$/Lb':<8} | {'%':<5} | {'$/Unit':<10}")
    print("-" * 110)

    # Mappings (Item, Percentage)
    mappings = [
        ('Nitrogen (Total/Urea)', 'NPKUREA', 46),
        ('Nitrogen (Ammoniacal)', 'NPKAN', 21),
        ('Nitrogen (Nitrate)', 'NO3CA', 11), # Using Calcium Nitrate 11%
        ('Phosphorus (Available)', 'NPKPHOS75', 54), # 75% H3PO4 approx 54% P2O5
        ('Potassium (Nitrate)', 'NPKKNO3', 45), # 13-0-45
        ('Magnesium', 'NO3MG63', 6.3),
        ('Manganese', 'NO3MN', 15),
        ('Copper', 'NO3CU', 14),
        ('Zinc', 'NO3ZN', 17),
        ('Iron', 'NO3FE', 10),
        ('Sulfur', 'CHEH2SO4', 30.3), # 93% H2SO4 approx 30.3% S
        ('Calcium', 'NO3CA', 19), # Ca in CaNO3? CN9 is usually 19% Ca. "11% / CN9". 
                                  # If 5Ca(NO3)2.NH4NO3.10H2O: 15.5% N, 19% Ca. 
                                  # Here desc says "11%". Let's assume the percentage in desc refers to N.
                                  # But for Calcium value, we need Ca %.
                                  # Pure Ca(NO3)2 is 24% Ca. 
                                  # Liquid CN9 is often 11% N? 
                                  # 'Calcium Nitrate 11% / CN9'. Usually liquid is 9-0-0-11Ca. 
                                  # But this is lbs. If dry, maybe 15.5-0-0-19Ca.
                                  # Let's check 'NO3CA12' Calcium Nitrate 12%. 
                                  # Let's assume standard 15.5-0-0-19Ca for dry if cost matches.
                                  # cost 0.1248/lb -> $250/ton. Liquid is cheaper. This might be liquid.
                                  # If liquid 11% N, it might be ~15% Ca?
                                  # Let's assume the percentage passed in tuple is the nutrient being valued.
    ]

    for category, item_code, percentage in mappings:
        if item_code in prices:
            p = prices[item_code]
            unit_val = calculate_unit_value(p['cost_lb'], percentage)
            print(f"{category:<30} | {item_code:<15} | {p['desc']:<30} | {p['cost_lb']:<8.4f} | {percentage:<5} | ${unit_val:<9.2f}")
        else:
            print(f"{category:<30} | {item_code:<15} | NOT FOUND")

if __name__ == "__main__":
    main()
