import fitz

doc = fitz.open(r'EMS(Levered)_updated_v3 (1).pdf')
print(f"PDF has {len(doc)} pages")

all_text = ""
for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        print(f"\n=== PAGE {i+1} ===")
        print(text[:5000])
        all_text += text

# Save to file
with open("pdf_content.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
print("\n\nSaved full content to pdf_content.txt")
