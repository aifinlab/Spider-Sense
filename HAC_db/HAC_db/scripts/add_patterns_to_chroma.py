import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
import os
import sys
import re
import json

# Add current directory to path to import vectorstore
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.vectorstore import VectorStore

PATTERNS_FILE = "/Users/jinzhiheng/Documents/智库/攻击数据集/数据清洗/final_add_attack_pattern.json"
PROMPTS_FILE = "/Users/jinzhiheng/Documents/智库/攻击数据集/数据清洗/stage_1_prompts.txt"

def parse_patterns_file(filepath):
    """
    Parses the file with format:
    # id='5' 
    { JSON }
    # id='6'
    { JSON }
    ...
    Returns a dict: {id_str: pattern_text}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by lines starting with #
    # Regex for split: `^#\s*id\s*=\s*[‘'](\d+)[’']`
    # Note: user file uses different quotes `‘` and `'` and maybe `’`.
    
    pattern_dict = {}
    
    # We'll stick to a simpler parsing: find all indices of lines starting with # id=
    # and extract the json block between them.
    
    # Normalize quotes for easier regex
    content = content.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')
    
    # Find all matches of ID lines
    # Pattern: # id='(\d+)' or # ID='(\d+)' or # id = '(\d+)'
    matches = list(re.finditer(r"#\s*(?:id|ID)\s*=\s*'(\d+)'", content))
    
    for i, match in enumerate(matches):
        id_val = match.group(1)
        start_idx = match.end()
        
        # End index is the start of next match or EOF
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
        else:
            end_idx = len(content)
            
        json_block = content[start_idx:end_idx].strip()
        
        try:
            data = json.loads(json_block)
            if 'pattern' in data:
                pattern_dict[id_val] = data['pattern']
            else:
                print(f"[!] Warning: No 'pattern' field for ID {id_val}")
        except json.JSONDecodeError as e:
            print(f"[!] Error decoding JSON for ID {id_val}: {e}")
            # Try to fix "}. " ending issue seen in file content?
            # The file ends with "}.", so maybe stripped ".".
            clean_block = json_block.strip()
            if clean_block.endswith('.'):
                clean_block = clean_block[:-1]
            try:
                data = json.loads(clean_block)
                if 'pattern' in data:
                    pattern_dict[id_val] = data['pattern']
            except:
                print(f"[!] Failed to parse JSON block for ID {id_val}")

    return pattern_dict

def parse_prompts_file(filepath):
    """
    Parses stage_1_prompts.txt to get ID -> prompt mapping.
    Format: === ID: 6 (...) ===\nContent...
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    parts = re.split(r'(=== ID: .*? ===\n)', content)
    prompt_dict = {}
    
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip()
        
        # Extract ID. Header: === ID: 6 (Most Dangerous: ...) ===
        # Or: === ID: 13 ===
        # We need the number. 
        # Match "ID: (\d+)"
        id_match = re.search(r"ID:\s*(\d+)", header)
        if id_match:
            id_val = id_match.group(1)
            prompt_dict[id_val] = body
            
    return prompt_dict

def main():
    print("[-] Parsing patterns...")
    patterns_map = parse_patterns_file(PATTERNS_FILE)
    print(f"[-] Loaded {len(patterns_map)} patterns: {list(patterns_map.keys())}")
    
    print("[-] Parsing prompts...")
    prompts_map = parse_prompts_file(PROMPTS_FILE)
    print(f"[-] Loaded {len(prompts_map)} prompts: {list(prompts_map.keys())}")
    
    # Match them
    texts_to_add = []
    metadatas_to_add = []
    
    for id_val, pattern_text in patterns_map.items():
        if id_val in prompts_map:
            prompt_text = prompts_map[id_val]
            texts_to_add.append(pattern_text)
            metadatas_to_add.append({"prompt": prompt_text, "original_id": id_val})
        else:
            print(f"[!] Warning: No matching prompt found for Pattern ID {id_val}")

    if not texts_to_add:
        print("[!] No matching data to add.")
        return

    print(f"[-] Adding {len(texts_to_add)} documents to ChromaDB...")
    
    # Initialize VectorStore
    # Assuming VectorStore class handles config internally (env extraction model etc)
    # Check build_chroma_db.py logic - it just inits VectorStore().
    try:
        vectorstore = VectorStore()
        vectorstore.add_documents(texts=texts_to_add, metadatas=metadatas_to_add)
        print(f"[+] Successfully added data to {vectorstore.persist_directory}")
    except Exception as e:
        print(f"[!] Error interacting with ChromaDB: {e}")

if __name__ == "__main__":
    main()
