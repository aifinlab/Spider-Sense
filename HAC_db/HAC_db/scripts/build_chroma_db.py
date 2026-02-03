import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
import os
import json
import argparse
from core.vectorstore import VectorStore

def build_database(dataset_path: str, reset: bool = False):
    """
    Reads a JSON dataset and adds valid entries to the ChromaDB vector store.
    
    Args:
        dataset_path (str): Path to the JSON file containing the dataset.
        reset (bool): If True, clears the existing database before adding new data.
    """
    print(f"[-] Initializing VectorStore...")
    # Initialize the VectorStore class (this handles the ChromaDB connection)
    vectorstore = VectorStore()
    
    if reset:
        print(f"[!] Warning: --reset flag is set. Clearing existing database at {vectorstore.persist_directory}...")
        vectorstore.clear_data()
        # Re-initialize explicitly or just rely on the empty dir, but VectorStore structure keeps the obj.
        # Since clear_data removes the dir, Chroma might need re-init in some versions, 
        # but usually performing add operations will re-create files.
        # To be safe, let's re-instantiate or just proceed if the lib handles it. 
        # Looking at vectorstore.py, it just does shutil.rmtree. 
        # Chroma client usually keeps an in-memory connection. 
        # Safest is to re-init VectorStore after clearing if the client invalidates.
        # But let's try simple clear first.

    
    print(f"[-] Loading data from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"[!] Error: File not found at {dataset_path}")
        return

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Error reading JSON file: {e}")
        return

    patterns = []
    prompts = []
    
    print(f"[-] Processing {len(data)} items...")
    
    for i, item in enumerate(data):
        # We need both 'pattern' (the essence) and 'adversarial' (the prompt)
        # Note: 'adversarial' is the key used in offline_essense_extraction.py, 
        # but sometimes it might be 'prompt'. We check both.
        
        essence = item.get('pattern')
        prompt = item.get('adversarial') or item.get('prompt')
        
        if essence and prompt:
            patterns.append(essence)
            prompts.append(prompt)
        else:
            # Optional: Log skipped items
            # print(f"[!] Skipping item {i}: Missing pattern or prompt")
            pass

    if not patterns:
        print("[!] No valid items found to add (Requires 'pattern' and 'adversarial'/'prompt' fields).")
        return

    print(f"[-] Found {len(patterns)} valid items to index.")
    
    # Create metadata list as required by VectorStore
    metadatas = [{"prompt": p} for p in prompts]
    
    # Add to database
    print(f"[-] Adding documents to ChromaDB...")
    vectorstore.add_documents(texts=patterns, metadatas=metadatas)
    
    print(f"[+] Database build complete! Data saved to: {vectorstore.persist_directory}")

if __name__ == "__main__":
    # You can run this script directly
    # Usage: python build_chroma_db.py --data /path/to/your/extracted_data.json
    
    parser = argparse.ArgumentParser(description="Build ChromaDB from extracted essence JSON.")
    parser.add_argument("--data", type=str, required=True, help="Path to the JSON data file")
    parser.add_argument("--reset", action="store_true", help="Clear the existing database before building (Create New). Default is Append.")
    
    args = parser.parse_args()
    
    build_database(args.data, args.reset)
