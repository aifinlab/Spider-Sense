# utils.py
import json
import os
import re
def append_to_json(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)

    with open(file_path, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    existing_data.append(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


import re
def filter_json(result):
    if not result:
        return None
    try:
        # 1. Try direct parsing
        return json.loads(result)
    except json.JSONDecodeError:
        pass

    try:
        # 2. Try extracting from code block
        match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        # 3. Try finding the first '{' and last '}'
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            return json.loads(match.group(0))
            
    except Exception as e:
        print(f"Warning: JSON extraction failed: {e}")
        
    return None