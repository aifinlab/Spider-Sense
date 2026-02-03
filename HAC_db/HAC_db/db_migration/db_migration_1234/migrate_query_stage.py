
import os
import sys
import json
import chromadb
from tqdm import tqdm

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = EDDF/EDDF/db_migration
# project_root should be EDDF/EDDF where 'core' folder resides
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from core.config import Config
from core.embedding import get_embedding_model

# Paths
INPUT_FILE = os.path.join(current_dir, "gpt4o_extracted_patterns.json")
# Define the new DB path explicitly as requested
# User request: /Users/jinzhiheng/Documents/agentsafe_project/EDDF/chroma/bge-m3/pre-planning
DB_NAME = "pre-planning"
DB_PATH = os.path.join(Config.BASE_DB_DIRECTORY, DB_NAME)

def migrate_pre_planning():
    print(f"[-] Starting migration for Pre-Planning Data...")
    print(f"[-] Input file: {INPUT_FILE}")
    print(f"[-] Target DB Path: {DB_PATH}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Input file not found: {INPUT_FILE}")
        return

    # 1. Load Data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[-] Loaded {len(data)} records.")

    # 2. Initialize ChromaDB
    # Ensure directory exists
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH, exist_ok=True)
        
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 3. Initialize Embedding Function
    print(f"[-] Initializing Embedding Model: {Config.EMBEDDING_MODEL_NAME} (BGE-M3)...")
    embedding_model = get_embedding_model(Config.EMBEDDING_MODEL_NAME)
    
    # Define a wrapper class to make it compatible with Chroma's embedding_function expectation if needed
    # Or simply compute embeddings first and pass them.
    # It's usually faster/stable to compute embeddings in batches and pass them to collection.add
    
    # 4. Create or Get Collection
    collection_name = "langchain"
    # Delete existing collection if we want a fresh start? 
    # The user implies a fresh creation "create a new folder... to store new data".
    try:
        client.delete_collection(collection_name)
        print(f"[-] Deleted existing collection: {collection_name}")
    except:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # BGE-M3 often uses cosine similarity
    )
    print(f"[-] Created collection: {collection_name}")

    # 5. Process and Batch Insert
    batch_size = 100
    
    ids_batch = []
    documents_batch = []
    metadatas_batch = []
    embeddings_batch = []
    
    counter = 0
    total = len(data)
    
    pbar = tqdm(total=total, desc="Migrating")
    
    # Use index + 1 for new sequential IDs
    for idx, item in enumerate(data):
        new_id = str(idx + 1)
        
        # Extract fields
        original_id = item.get("id", "")
        original_prompt = item.get("original_prompt", "")
        pattern = item.get("pattern", "")
        extracted_result = item.get("extracted_result", {})
        
        # Prepare components string
        components_str = ""
        if isinstance(extracted_result, dict) and "components" in extracted_result:
            components_str = json.dumps(extracted_result["components"], ensure_ascii=False)
            
        full_extracted_result_str = json.dumps(extracted_result, ensure_ascii=False)
        
        # Ensure pattern is not empty for embedding
        doc_content = pattern
        if not doc_content:
            # Fallback if pattern is missing ? (Should not happen ideally)
            # Use original prompt or just skip? 
            # If skip, ID sequence breaks. Let's use empty string place holder or log warning
            print(f"[!] Warning: Record {original_id} has empty pattern.")
            doc_content = " " 
        
        # Prepare Metadata
        # Chroma metadata values must be int, float, str, or bool. No lists/dicts.
        metadata = {
            "original_id": original_id,
            "original_prompt": original_prompt[:5000],  # Truncate if too long to ensure no metadata size issues?
            # actually original_prompt can be very long. Chroma usually handles it but let's be safe if it's huge.
            # But user wants "don't upload duplicate fields but ensure ALL fields uploaded".
            # We'll try to keep full text.
            "components": components_str,
            "extracted_result": full_extracted_result_str
        }
        
        # Compute Embedding for this doc
        # Doing one by one is slow, but simpler to correct logic. 
        # Better: Accumulate text and batch embed.
        
        ids_batch.append(new_id)
        documents_batch.append(doc_content)
        metadatas_batch.append(metadata)
        
        if len(ids_batch) >= batch_size:
            # Compute embeddings
            embeddings = embedding_model.embed_documents(documents_batch)
            
            # Add to collection
            collection.add(
                ids=ids_batch,
                embeddings=embeddings,
                metadatas=metadatas_batch,
                documents=documents_batch
            )
            
            # Reset batch
            ids_batch = []
            documents_batch = []
            metadatas_batch = []
            embeddings_batch = []
            
        pbar.update(1)
        
    # Process remaining
    if ids_batch:
        embeddings = embedding_model.embed_documents(documents_batch)
        collection.add(
            ids=ids_batch,
            embeddings=embeddings,
            metadatas=metadatas_batch,
            documents=documents_batch
        )
        
    pbar.close()
    
    print(f"[+] Migration complete.")
    print(f"[-] Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    migrate_pre_planning()
