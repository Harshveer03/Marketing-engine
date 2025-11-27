import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Import your existing modules
from simple_embedder import create_embeddings
from extractor import extract_structured_info, load_faiss_index, query_icp

load_dotenv()

def main():
    print("🎯 Marketing Engine Setup")
    print("=" * 40)
    
    # Check if already configured
    if os.path.exists("./config/is_configured.flag"):
        print("✅ Already configured!")
        reconfigure = input("Reconfigure with new document? (y/n): ")
        if reconfigure.lower() != 'y':
            print("Setup cancelled. Run 'python app.py' to start the dashboard.")
            return
    
    # Get business document
    pdf_path = get_business_document()
    
    # Process document
    print("\n🔄 Processing your business document...")
    try:
        create_embeddings()
        print("✅ Document processed and embedded successfully!")
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return
    
    print("🎯 Extracting your business niche...")
    try:
        vectordb = load_faiss_index()
        context = query_icp(vectordb)
        niche_data = extract_structured_info(context)
        
        # Save to generated folder
        os.makedirs("./generated", exist_ok=True)
        with open("./generated/niche_icp.json", "w", encoding="utf-8") as f:
            json.dump(niche_data, f, indent=4, ensure_ascii=False)
        
        print("✅ Business niche extracted successfully!")
    except Exception as e:
        print(f"❌ Error extracting niche: {e}")
        return
    
    # Save configuration
    config = {
        "business_name": niche_data.get("industry", "Your Business"),
        "setup_date": datetime.now().isoformat(),
        "document_processed": pdf_path,
        "target_audience": niche_data.get("target_audience", []),
        "value_proposition": niche_data.get("value_proposition", "")
    }
    
    os.makedirs("./config", exist_ok=True)
    with open("./config/settings.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create necessary directories
    directories = [
        "./generated/content/blogs",
        "./generated/content/social",
        "./generated/analytics",
        "./generated/news",
        "./generated/topics"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Mark as configured
    with open("./config/is_configured.flag", "w") as f:
        f.write("configured")
    
    print("\n✅ Setup Complete!")
    print("🚀 Your marketing engine is ready!")
    print("📊 Run 'python app.py' to start the web dashboard!")
    print("💡 Or run 'python marketing_engine.py' for CLI interface")

def get_business_document():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    while True:
        files = [f for f in os.listdir(data_dir) 
                if f.endswith(('.pdf', '.docx', '.txt'))]
        
        if not files:
            print(f"\n📁 Please place your business document in: {data_dir}")
            print("Supported: PDF, DOCX, TXT files")
            input("Press Enter after adding your file...")
            continue
        
        if len(files) == 1:
            selected_file = os.path.join(data_dir, files[0])
            print(f"📄 Found document: {files[0]}")
            return selected_file
        
        # Multiple files - let user choose
        print("\nMultiple documents found:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input("Select file number: ")) - 1
            if 0 <= choice < len(files):
                return os.path.join(data_dir, files[choice])
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")

if __name__ == "__main__":
    main()