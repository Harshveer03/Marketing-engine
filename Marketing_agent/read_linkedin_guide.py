from pypdf import PdfReader

pdf_path = "Engine rules/01-Content-Generation/Linkedin Content Guide.pdf"
reader = PdfReader(pdf_path)

print(f"📄 Reading: {pdf_path}")
print(f"📊 Total pages: {len(reader.pages)}\n")

text = ""
for i, page in enumerate(reader.pages, 1):
    page_text = page.extract_text()
    text += page_text + "\n"
    print(f"Page {i}: {len(page_text)} characters")

print(f"\n✅ Total extracted: {len(text)} characters")
print(f"\n{'='*60}")
print("CONTENT PREVIEW:")
print(f"{'='*60}")
print(text[:2000])
print("...")
