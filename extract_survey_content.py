import fitz

pdf_path = r'c:\Users\alexh\Downloads\mod\Fertilizer Commercial Values Survey 25-26.pdf'
print(f"Opening {pdf_path}")

try:
    doc = fitz.open(pdf_path)
    print(f"PDF has {len(doc)} pages")

    all_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"\n=== PAGE {i+1} ===")
        print(text)
        all_text += text

except Exception as e:
    print(f"Error: {e}")
