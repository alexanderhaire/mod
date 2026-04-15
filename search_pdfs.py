import fitz
import glob
import os

pdf_files = glob.glob(r'c:\Users\alexh\Downloads\mod\*.pdf')

search_terms = ['Sucrate', 'Methylene', 'IBDU', 'Resin', 'Polyon', 'Coated', 'Dolomite', 'Limestone']

for pdf in pdf_files:
    if 'Filled' in pdf or 'Fertilizer Commercial Values Survey' in pdf:
        continue
    
    print(f"\n--- Searching {os.path.basename(pdf)} ---")
    try:
        doc = fitz.open(pdf)
        found = False
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            for lines in text.split('\n'):
                for term in search_terms:
                    if term.lower() in lines.lower():
                        print(f"Page {page_num+1}: {lines.strip()}")
                        found = True
        
        if not found:
            print("No matches found.")
    except Exception as e:
        print(f"Error reading {pdf}: {e}")
