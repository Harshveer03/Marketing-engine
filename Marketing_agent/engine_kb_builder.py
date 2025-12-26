"""
Engine Knowledge Base Builder
Processes MIMIR rules and creates a searchable vector database
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def create_engine_kb(
    documents_path="./Engine rules/01-Content-Generation",
    output_path="./engine_kb/vectordb",
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Process Engine Rules PDFs and create vector database
    
    Args:
        documents_path: Path to organized Engine Rules
        output_path: Where to save the vector database
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Overlap between chunks
    
    Returns:
        FAISS vector database
    """
    print("🔧 Building Engine Knowledge Base...\n")
    
    # Collect all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(documents_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    # Load all documents
    documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"📄 Loading: {filename}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Add metadata to each document
            for doc in docs:
                doc.metadata['source_file'] = filename
                doc.metadata['category'] = 'content_generation'
                
                # Add specific metadata based on filename
                if 'MIMIR' in filename:
                    doc.metadata['type'] = 'system'
                elif 'Part-' in filename:
                    part_num = filename.split('-')[1]
                    doc.metadata['type'] = 'part'
                    doc.metadata['part_number'] = part_num
            
            documents.extend(docs)
            print(f"   ✅ Loaded {len(docs)} pages")
            
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    print(f"\n📊 Total pages loaded: {len(documents)}")
    
    # Split documents into chunks
    print(f"\n✂️ Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"   ✅ Created {len(splits)} chunks")
    
    # Create embeddings
    print(f"\n🧠 Creating embeddings using Ollama...")
    print(f"   This may take 2-3 minutes for {len(splits)} chunks...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create vector store
    print(f"💾 Building FAISS vector database...")
    print(f"   Processing embeddings...")
    vectordb = FAISS.from_documents(splits, embeddings)
    print(f"   ✅ Vector database created!")
    
    # Save to disk
    os.makedirs(output_path, exist_ok=True)
    vectordb.save_local(output_path)
    
    print(f"\n✅ Engine KB created successfully!")
    print(f"   Location: {output_path}")
    print(f"   Total chunks: {len(splits)}")
    print(f"   Source files: {len(pdf_files)}")
    
    return vectordb

def test_engine_kb(vectordb_path="./engine_kb/vectordb"):
    """
    Test the Engine KB with sample queries
    """
    print("\n🧪 Testing Engine KB...\n")
    
    # Load the vector database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.load_local(
        vectordb_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Test queries
    test_queries = [
        "How to handle tone for professional B2B audience",
        "Visual orchestration rules for image generation",
        "Persona tailoring principles",
        "Anti-patterns to avoid in content",
        "Grounding and integrity rules"
    ]
    
    for query in test_queries:
        print(f"🔍 Query: '{query}'")
        docs = vectordb.similarity_search(query, k=2)
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            print(f"   {i}. {source}")
            print(f"      {doc.page_content[:150]}...")
        print()
    
    print("✅ Engine KB test complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        test_engine_kb()
    else:
        # Build mode
        vectordb = create_engine_kb()
        
        print("\n" + "="*60)
        print("Next steps:")
        print("1. Test the KB: python engine_kb_builder.py test")
        print("2. Use engine_kb_helper.py to query rules during content generation")
        print("="*60)
