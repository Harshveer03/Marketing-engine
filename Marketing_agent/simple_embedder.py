import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

def create_embeddings():
    """Simple function to create embeddings from PDFs in data folder"""
    
    PDF_DIR = "./data"
    VECTOR_DB_DIR = "./vectordb"
    RECORD_FILE = "embedding_record.json"
    
    print("🔄 Creating embeddings from your document...")
    
    # Initialize embeddings
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        raise Exception("No PDF files found in data directory")
    
    # Load documents
    all_docs = []
    for pdf_file in pdf_files:
        print(f"📄 Processing {pdf_file}...")
        loader = PyPDFLoader(os.path.join(PDF_DIR, pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_file
        all_docs.extend(docs)
    
    print(f"📑 Loaded {len(all_docs)} pages")
    
    # Split documents
    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(all_docs)
    print(f"🔖 Created {len(docs)} chunks")
    
    # Create vector store
    print("🗂️ Creating vector database...")
    vectordb = FAISS.from_documents(docs, embedding)
    
    # Save vector store
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectordb.save_local(VECTOR_DB_DIR)
    
    # Update record
    record = {}
    for pdf_file in pdf_files:
        record[pdf_file] = os.path.getmtime(os.path.join(PDF_DIR, pdf_file))
    
    with open(RECORD_FILE, "w") as f:
        json.dump(record, f, indent=2)
    
    print("✅ Embeddings created successfully!")
    return True

if __name__ == "__main__":
    create_embeddings()