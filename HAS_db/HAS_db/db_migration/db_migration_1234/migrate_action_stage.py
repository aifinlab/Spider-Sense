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
from core.prompts import pattern_judge_prompt_stage_2
from core.utils import filter_json
from core.vectorstore import VectorStore
from core.embedding import BGEM3Embedding

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

def parse_stage_2_cases(filepath):
    """
    Parses the text file into individual cases with structured fields.
    Returns a list of dictionaries.
    """
    content = read_file_content(filepath)
    
    # Split by "User Instruction:" to identify case blocks
    # We use lookahead or just standard split and reconstruct
    parts = re.split(r"(?:\n|^)User Instruction:", content)
    
    cases = []
    for part in parts:
        if not part.strip():
            continue
        
        # Check if this part contains the expected fields
        if "Agent Plan:" in part and "Tool Information:" in part:
            # Reconstruct the full text for the scenario field (legacy/fallback)
            full_text = "User Instruction:" + part
            
            # Extract fields using regex
            # Note: We append the delimiter "User Instruction:" back effectively by matching from start of 'part' 
            # after we split it off. But 'part' starts with value of User Instruction.
            
            # Let's parse 'part' which is everything AFTER "User Instruction:"
            
            user_instruction_match = re.match(r"(.*?)(?:\nAgent Plan:|$)", part, re.DOTALL)
            user_instruction = user_instruction_match.group(1).strip() if user_instruction_match else ""
            
            agent_plan_match = re.search(r"Agent Plan:(.*?)(?:\nTool Information:|$)", part, re.DOTALL)
            agent_plan = agent_plan_match.group(1).strip() if agent_plan_match else ""
            
            tool_info_match = re.search(r"Tool Information:(.*?)(?:\nTool Call Draft:|$)", part, re.DOTALL)
            tool_info = tool_info_match.group(1).strip() if tool_info_match else ""
            
            tool_draft_match = re.search(r"Tool Call Draft:(.*?)(?:\nApproval Reason/Constraint:|$)", part, re.DOTALL)
            tool_draft = tool_draft_match.group(1).strip() if tool_draft_match else ""
            
            approval_reason_match = re.search(r"Approval Reason/Constraint:(.*?)(?:\n|$)", part, re.DOTALL)
            approval_reason = approval_reason_match.group(1).strip() if approval_reason_match else ""
            
            cases.append({
                "full_text": full_text.strip(),
                "user_instruction": user_instruction,
                "agent_plan": agent_plan,
                "tool_information": tool_info,
                "tool_call_draft": tool_draft,
                "approval_reason": approval_reason
            })
             
    return cases

def migrate_stage_2():
    print("[-] Starting Stage 2 Migration...")
    
    # 1. Setup Directories
    INPUT_DIR_OUTPUTS = "/Users/jinzhiheng/Documents/agentsafe_project/数据清洗/stage_2_llm_outputs"
    INPUT_FILE_CASES = "/Users/jinzhiheng/Documents/agentsafe_project/数据清洗/stage_2_llm_inputs/阶段二攻击案例en.txt"
    DB_PATH = Config.DB_PATH_PRE_ACTION
    
    print(f"[-] Inputs File: {INPUT_FILE_CASES}")
    print(f"[-] Outputs Dir: {INPUT_DIR_OUTPUTS}")
    print(f"[-] Target DB Path: {DB_PATH}")
    
    if not os.path.exists(INPUT_FILE_CASES):
        print(f"[!] Error: Input cases file not found: {INPUT_FILE_CASES}")
        return
        
    if not os.path.exists(INPUT_DIR_OUTPUTS):
        print(f"[!] Error: Output directory not found: {INPUT_DIR_OUTPUTS}")
        return

    # 2. Parse Cases
    cases = parse_stage_2_cases(INPUT_FILE_CASES)
    print(f"[-] Parsed {len(cases)} cases from text file.")
    
    # 3. Setup Client and VectorStore
    client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_API_BASE)
    
    original_persist_dir = Config.PERSIST_DIRECTORY
    Config.PERSIST_DIRECTORY = DB_PATH
    print(f"[-] Set ChromaDB persist directory to: {Config.PERSIST_DIRECTORY}")
    
    try:
        vectorstore = VectorStore(db_type="pre_action")
    except Exception as e:
        print(f"[!] Error initializing VectorStore: {e}")
        return

    # 4. Processing Loop (Stitching)
    processed_count = 0
    added_count = 0
    skipped_judge_count = 0
    failed_count = 0
    
    # We expect 1-to-1 mapping: case[0] -> stage_2_pattern_1.json, case[1] -> stage_2_pattern_2.json ...
    
    for i, case_data in enumerate(tqdm(cases, desc="Processing cases")):
        case_id = i + 1
        json_filename = f"stage_2_pattern_{case_id}.json"
        json_path = os.path.join(INPUT_DIR_OUTPUTS, json_filename)
        
        if not os.path.exists(json_path):
            print(f"[!] Warning: Missing JSON file for case {case_id}: {json_filename}")
            failed_count += 1
            continue
            
        try:
            # Read Pattern Data
            with open(json_path, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)
            
            # Prepare Judge Prompt
            # case_content is the "Scenario Input"
            
            components = json.dumps(pattern_data.get("components", []), ensure_ascii=False)
            pattern = pattern_data.get("pattern", "")
            
            if not pattern:
                print(f"[!] Warning: No pattern found in {json_filename}")
                failed_count += 1
                continue

            judge_prompt = pattern_judge_prompt_stage_2.format(
                scenario_input=case_data["full_text"],
                components=components,
                pattern=pattern
            )
            
            # Run Judge
            judge_result_str = run_llm(judge_prompt, client)
            try:
                judge_result = filter_json(judge_result_str)
            except Exception as e:
                print(f"[!] Error parsing judge result for case {case_id}: {e}")
                failed_count += 1
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
                # Structured Metadata
                metadata = {
                    "original_id": f"stage_2_{case_id}",
                    "attack_category": pattern_data.get("attack_category", "Unknown"),
                    # Structured Fields
                    "user_instruction": case_data["user_instruction"],
                    "agent_plan": case_data["agent_plan"],
                    "tool_information": case_data["tool_information"],
                    "tool_call_draft": case_data["tool_call_draft"],
                    "approval_reason": case_data["approval_reason"]
                }
                
                vectorstore.add_documents(texts=[pattern], metadatas=[metadata])
                added_count += 1
            else:
                skipped_judge_count += 1
                # print(f"[-] Judge failed for case {case_id}")
                
            processed_count += 1
            
        except Exception as e:
            print(f"[!] Error processing case {case_id}: {e}")
            failed_count += 1

    print("-" * 30)
    print(f"Migration Complete.")
    print(f"Processed: {processed_count}")
    print(f"Added to DB: {added_count}")
    print(f"Skipped (Judge Failed): {skipped_judge_count}")
    print(f"Failed (Errors): {failed_count}")
    
    Config.PERSIST_DIRECTORY = original_persist_dir

if __name__ == "__main__":
    migrate_stage_2()
