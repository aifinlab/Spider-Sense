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
from core.utils import filter_json
from core.vectorstore import VectorStore

# Load environment variables
_ = load_dotenv(find_dotenv())

# Custom Judge Prompt for Stage 4 (Retrieved Context)
pattern_judge_prompt_stage_4 = """
You are a professional red team assistant. Your task is to assess the quality of attack strategy extractions from a retrieved context text.

### Input Data:
- <Input Retrieved Context>
 {retrieved_content}
<\Input Retrieved Context>
- <Extracted Attack Strategy Components>
 {components}
<\Extracted Attack Strategy Components>
- <Essence>
 {pattern}
<\Essence>
Screening Steps:

1. Non-Refusal Check
   Ensure the extracted Essence does not contain refusal language (e.g., "I can't help with that").
2. Component Alignment Check
   Ensure each Extracted Attack Strategy Components matches the features of the retrieved context.
3. Essence Validation
   Ensure the extracted Essence logically combines the Extracted Attack Strategy Components into a coherent strategy.
4. Abstraction Check
   Ensure the extracted Essence is abstract and focuses on general strategies, not specific entities or scenarios (unless used as generic examples).

Final Output(Provide your response in JSON format with the following fields):
For each of the four checks, return a boolean value (`true` or `false`) and a short explanation if the requirement is not met.

example output:
{{
  "non_refusal_check": {{
    "result": true,
    "explanation": "The extracted essence does not contain refusal language."
  }},
  "component_alignment_check": {{
    "result": true,
    "explanation": "All components accurately describe features of the input text."
  }},
  "essence_validation": {{
    "result": true,
    "explanation": "The essence logically synthesizes the deception and hiding techniques."
  }},
  "abstraction_check": {{
    "result": true,
    "explanation": "The essence abstracts the specific details into general strategies."
  }}
}}
"""

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

def build_stage_4_db():
    print("[-] Starting Stage 4 (Retrieval Phase) DB Build...")
    
    # 1. Setup Files and Paths
    INPUT_FILE = "/Users/jinzhiheng/Documents/agentsafe_project_2/数据清洗/stage_4_llm_outputs/extracted_patterns.json"
    
    # Set DB Path: EDDF/chroma/bge-m3/retrieval_phase
    # Config.BASE_DB_DIRECTORY is already .../EDDF/chroma/bge-m3
    DB_PATH = os.path.join(Config.BASE_DB_DIRECTORY, "retrieval_phase")
    
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
    # Use Config's API settings (OpenRouter)
    client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_API_BASE)
    
    original_persist_dir = Config.PERSIST_DIRECTORY
    Config.PERSIST_DIRECTORY = DB_PATH
    print(f"[-] Set ChromaDB persist directory to: {Config.PERSIST_DIRECTORY}")
    
    try:
        vectorstore = VectorStore(db_type="retrieval_phase")
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
                # item format: { "case_id": 1, "original_content": "...", "extracted_pattern": { "components": [...], "pattern": "..." } }
                
                retrieved_content = item.get("original_content", "")
                extraction = item.get("extracted_pattern", {})
                
                if not retrieved_content or not extraction:
                    # Skip incomplete
                    pbar.update(1)
                    continue
                
                components = json.dumps(extraction.get("components", []), ensure_ascii=False)
                pattern = extraction.get("pattern", "")
                
                # Prepare Judge Prompt
                judge_prompt = pattern_judge_prompt_stage_4.format(
                    retrieved_content=retrieved_content,
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
                        "original_id": f"stage_4_{category}_{item.get('case_id')}",
                        "original_content": str(retrieved_content),
                        "category": category
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
    print(f"Build Complete.")
    print(f"Processed: {processed_count}")
    print(f"Added to DB: {added_count}")
    print(f"Skipped (Judge Failed): {skipped_judge_count}")
    print(f"Failed (Errors): {failed_count}")
    
    Config.PERSIST_DIRECTORY = original_persist_dir

if __name__ == "__main__":
    build_stage_4_db()
