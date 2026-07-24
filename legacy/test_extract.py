import fitz  # PyMuPDF

pdf_path = "sample.pdf"

doc = fitz.open(pdf_path)

for page_num, page in enumerate(doc, start=1):
    text = page.get_text()
    print(f"\n--- Page {page_num} ---")
    print(text)