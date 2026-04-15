import fitz

pdf_paths = [
    r'C:\Users\alexh\Downloads\mod\AlphaMax_Pro_Report.pdf',
    r'C:\Users\alexh\Downloads\mod\ERP_Regime_Pro_Report.pdf'
]

for pdf_path in pdf_paths:
    output_path = pdf_path.replace('.pdf', '_content.txt')
    try:
        doc = fitz.open(pdf_path)
        print(f"Opening {pdf_path} with {len(doc)} pages")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(doc):
                text = page.get_text()
                f.write(f"\n=== PAGE {i+1} ===\n")
                f.write(text)
                
        print(f"Successfully saved content to {output_path}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
