import fitz

pdf_path = r'c:\Users\alexh\Downloads\mod\Fertilizer Commercial Values Survey 25-26.pdf'
out_path = r'c:\Users\alexh\Downloads\mod\Fertilizer Commercial Values Survey 25-26_Filled.pdf'
doc = fitz.open(pdf_path)

# Dictionary of terms to search for and the value to place next to it
values = {
    "Total Nitrogen": "13.54",
    "Nitrate Nitrogen": "27.78",
    "Ammoniacal Nitrogen": "23.47",
    "Water Soluble Nitrogen or Urea": "13.54",
    "Available Phosphorus": "15.99",
    "Potassium (from Muriate)": "6.69",
    "Potassium (from any source": "26.73",
    "Magnesium": "61.91",
    "Manganese (from sulfate)": "26.25",
    "Manganese (chloride)": "56.19",
    "Manganese (from sucrate)": "120.02", # New
    "Manganese (from oxide)": "16.04", 
    "Manganese (from chelate in group 1": "314.41",
    "Copper (from sulfate)": "147.62",
    "Copper (from sucrate)": "223.53", # New
    "Copper (from chelate in group 1": "306.57",
    "Zinc (from chloride)": "48.46",
    "Zinc (from sucrate)": "68.90", # New
    "Zinc (from oxide)": "49.69", 
    "Zinc (from chelate in group 1": "190.82",
    "Iron (from sulfate)": "13.06",
    "Iron (from sucrate)": "80.43", # New - typically labeled 'Iron (from sucrate)' or 'Iron Sucrate'
    "Iron Humate": "103.84", # Updated from 1130.00
    "Iron (from chelate in group 1": "612.33",
    "Sulfur (combined - from Sulfates)": "20.46",
    "Boron": "168.10",
    "Cobalt": "716.38",
    "Molybdenum": "611.12",
    "Calcium (from any source)": "22.73",
    "Calcium Sulfate (Land Plaster, Gypsum)": "118.14",
    "Seaweed Extract": "47.60",
    "Urea Triazone": "26.67", 
}

x_pos_main = 465 # X coordinate for the '$' column
x_pos_slow = 370 # X coordinate for the Slow Release section ($ column)

for page_num in range(len(doc)):
    page = doc[page_num]
    for field, val in values.items():
        rects = page.search_for(field)
        if rects:
            for r in rects:
                if field == "Magnesium":
                    if page_num == 1: 
                        page.insert_text((x_pos_main, r.y1 - 2), str(val), fontsize=11, fontname="helv", color=(0, 0, 1))
                elif field == "Calcium (from any source)":
                    page.insert_text((x_pos_main, r.y1 - 2), str(val), fontsize=11, fontname="helv", color=(0, 0, 1))
                elif field == "Calcium Sulfate (Land Plaster, Gypsum)":
                     page.insert_text((x_pos_main, r.y1 - 2), str(val), fontsize=11, fontname="helv", color=(0, 0, 1))
                elif field == "Urea Triazone":
                     page.insert_text((x_pos_slow, r.y1 - 2), str(val), fontsize=11, fontname="helv", color=(0, 0, 1))
                else:
                    page.insert_text((x_pos_main, r.y1 - 2), str(val), fontsize=11, fontname="helv", color=(0, 0, 1))

doc.save(out_path)
print(f"Successfully saved to {out_path}")
