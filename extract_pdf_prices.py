import os
import re
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed. Please run: pip install pypdf")
    import sys
    sys.exit(1)

def extract_pdf_prices():
    print("--- QUANTITATIVE RESEARCH: PDF Price Scraping ---")
    
    # Files identified from list_dir
    pdf_files = [
        "Offer Sheet Glacial Acetic Acid SSi788.pdf",
        "Offer Sheet Hexane SSi779.pdf",
        "Offer Sheet KCL SSi784.pdf",
        "Offer Sheet Kristalex Hydrocarbon Resin SSi780.pdf",
        "Offer Sheet Magnesium Citrate Tribasic SSi787.pdf",
        "Offer Sheet Sodium Aluminate SSi789.pdf",
        "Offer Sheet Sulfamic SSi785.pdf",
        "Offer sheet Silica Gel SSi782.pdf",
        "Offer sheet TEA SSi768.pdf",
        "YNA EC Price List 1-5-2026.pdf"
    ]
    
    results = []
    
    for filename in pdf_files:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (Not found)")
            continue
            
        print(f"Scanning {filename}...")
        try:
            reader = PdfReader(filename)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Simple Regex for Price-like patterns associated with items
            # looking for lines with $ sign or numbers near keywords like "Price", "Cost", "lb", "kg"
            
            # Simple dump of lines with "$" signs
            lines = text.split('\n')
            for line in lines:
                if '$' in line or 'USD' in line or 'Price' in line:
                    clean_line = line.strip()
                    if len(clean_line) > 5: # Ignore noise
                        print(f"  > Found: {clean_line}")
                        results.append({'File': filename, 'Context': clean_line})
                        
        except Exception as e:
            print(f"  Error reading {filename}: {e}")

    if results:
        print(f"\nFound {len(results)} potential pricing lines.")
    else:
        print("No pricing patterns found in PDFs.")

if __name__ == "__main__":
    extract_pdf_prices()
