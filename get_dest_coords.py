import fitz

pdf_path = r'c:\Users\alexh\Downloads\mod\Fertilizer Commercial Values Survey 25-26.pdf'
doc = fitz.open(pdf_path)

fields = [
    "Total Nitrogen",
    "Nitrate Nitrogen",
    "Ammoniacal Nitrogen",
    "Water Soluble Nitrogen or Urea",
    "Available Phosphorus",
    "Potassium (from Muriate)",
    "Potassium (from any source"
]

for page_num in range(len(doc)):
    page = doc[page_num]
    for field in fields:
        rects = page.search_for(field)
        if rects:
            print(f"Found '{field}' on page {page_num + 1} at {rects[0]}")
