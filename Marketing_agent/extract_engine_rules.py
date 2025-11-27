"""
Extract text from Engine Rules PDFs
"""
import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def extract_all_pdfs(directory):
    """Extract text from all PDFs in directory"""
    results = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"📄 Extracting: {file}")
                
                text = extract_text_from_pdf(pdf_path)
                if text:
                    results[file] = text
                    print(f"   ✅ Extracted {len(text)} characters")
                else:
                    print(f"   ❌ Failed to extract")
    
    return results

if __name__ == "__main__":
    print("🔍 Extracting Engine Rules from PDFs...\n")
    
    engine_rules_dir = "./Engine rules/Content Generation Rules"
    
    if not os.path.exists(engine_rules_dir):
        print(f"❌ Directory not found: {engine_rules_dir}")
        exit(1)
    
    # Extract all PDFs
    extracted_texts = extract_all_pdfs(engine_rules_dir)
    
    print(f"\n📊 Summary:")
    print(f"   Total PDFs processed: {len(extracted_texts)}")
    
    # Save to text files for review
    output_dir = "./engine_rules_extracted"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, text in extracted_texts.items():
        output_file = os.path.join(output_dir, filename.replace('.pdf', '.txt'))
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"   💾 Saved: {output_file}")
    
    print(f"\n✅ All PDFs extracted to: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Review the extracted text files")
    print(f"2. Run engine_kb_builder.py to create the vector database")
