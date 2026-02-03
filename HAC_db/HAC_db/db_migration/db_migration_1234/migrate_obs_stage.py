import sys
import os
import json
import re
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config import Config
from core.prompts import pattern_judge_prompt_stage_3
from core.utils import filter_json
from core.vectorstore import VectorStore
# BGEM3Embedding is loaded inside VectorStore via Config

# Load environment variables
_ = load_dotenv(find_dotenv())

def run_llm(prompt: str, client: OpenAI) -> str:
    completion = client.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are a professional Red Team Assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        model=Config.model_name,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return completion.choices[0].message.content

def read_file_content(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def migrate_stage_3():
    print("[-] Starting Stage 3 Migration...")
    
    # 1. Setup Directories and Files
    INPUT_FILE = "/Users/jinzhiheng/Documents/agentsafe_project/数据清洗/stage_3_llm_outputs/stage_3_extracted_patterns.json"
    # User requested "post-observation" (hyphen) specifically for the new DB
    DB_PATH = os.path.join(Config.BASE_DB_DIRECTORY, "post-observation")
    
    print(f"[-] Input File: {INPUT_FILE}")
    print(f"[-] Target DB Path: {DB_PATH}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: Input file not found: {INPUT_FILE}")
        return

    # 2. Load Data
    try:
        data = json.loads(read_file_content(INPUT_FILE))
    except Exception as e:
        print(f"[!] Error loading input JSON: {e}")
        return

    # 3. Setup Client and VectorStore
    client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_API_BASE)
    
    original_persist_dir = Config.PERSIST_DIRECTORY
    Config.PERSIST_DIRECTORY = DB_PATH
    print(f"[-] Set ChromaDB persist directory to: {Config.PERSIST_DIRECTORY}")
    
    try:
        vectorstore = VectorStore(db_type="post_observation")
    except Exception as e:
        print(f"[!] Error initializing VectorStore: {e}")
        return

    # 4. Processing Loop
    # Data format: Dict[Category, List[Items]]
    
    processed_count = 0
    added_count = 0
    skipped_judge_count = 0
    failed_count = 0
    
    # Get total count for progress bar
    total_items = sum(len(items) for items in data.values())
    
    pbar = tqdm(total=total_items, desc="Processing items")
    
    for category, items in data.items():
        for item in items:
            try:
                # Extract fields
                # item format: { "case_id": 1, "original_tool_output": "...", "extraction": { "components": [...], "pattern": "..." } }
                
                tool_output = item.get("original_tool_output", "")
                extraction = item.get("extraction", {})
                
                if not tool_output or not extraction:
                    # Skip incomplete
                    pbar.update(1)
                    continue
                
                components = json.dumps(extraction.get("components", []), ensure_ascii=False)
                pattern = extraction.get("pattern", "")
                
                # Prepare Judge Prompt
                judge_prompt = pattern_judge_prompt_stage_3.format(
                    tool_output=tool_output,
                    components=components,
                    pattern=pattern
                )
                
                # Run Judge
                judge_result_str = run_llm(judge_prompt, client)
                try:
                    judge_result = filter_json(judge_result_str)
                except Exception as e:
                    print(f"[!] Error parsing judge result for ID {item.get('case_id')}: {e}")
                    failed_count += 1
                    pbar.update(1)
                    continue
                    
                # Check results
                checks = ["non_refusal_check", "component_alignment_check", "essence_validation", "abstraction_check"]
                all_passed = True
                for check in checks:
                    if check in judge_result and isinstance(judge_result[check], dict):
                         if not judge_result[check].get("result", False):
                             all_passed = False
                             break
                    else:
                         all_passed = False 
                         break
                
                if all_passed:
                    # Add to VectorDB
                    metadata = {
                        "original_id": f"stage_3_{category}_{item.get('case_id')}",
                        "tool_output": str(tool_output), # Key renamed as requested
                        "category": category
                        # "components": components
                    }
                    
                    vectorstore.add_documents(texts=[pattern], metadatas=[metadata])
                    added_count += 1
                else:
                    skipped_judge_count += 1
                
                processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"[!] Error processing item in {category}: {e}")
                failed_count += 1
                pbar.update(1)

    pbar.close()
    
    print("-" * 30)
    print(f"Migration Complete.")
    print(f"Processed: {processed_count}")
    print(f"Added to DB: {added_count}")
    print(f"Skipped (Judge Failed): {skipped_judge_count}")
    print(f"Failed (Errors): {failed_count}")
    
    Config.PERSIST_DIRECTORY = original_persist_dir

if __name__ == "__main__":
    migrate_stage_3()
