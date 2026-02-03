import sys
import os
import json
import re
from typing import Dict, Any, Tuple

# Ensure EDDF modules can be imported
# Assuming AgentSafe_1_workflow is a sibling of EDDF in agentsafe_project
# Current file: .../SpiderSense/pyopenagi/agents/sandbox.py
# Root of workflow: .../SpiderSense
current_file_path = os.path.abspath(__file__)
# Go up 3 levels: agents -> pyopenagi -> SpiderSense
workflow_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
project_root = workflow_root # Assuming SpiderSense contains everything, including EDDF if needed
eddf_inner_path = os.path.join(project_root, "EDDF", "EDDF")

if eddf_inner_path not in sys.path:
    sys.path.append(eddf_inner_path)

try:
    from core.vectorstore import VectorStore
    from core.config import Config
except ImportError:
    print(f"Warning: Could not import EDDF modules from {eddf_inner_path}. Vector features will be disabled.")
    VectorStore = None
    Config = None

class VectorFeedbackSandbox:
    def __init__(self, stage: int, llm_client=None):
        """
        Initialize the VectorFeedbackSandbox.
        
        Args:
            stage (int): The current stage of the agent (1, 2, 3, or 4).
            llm_client: Client to use for LLM calls (extraction/judge).
        """
        self.stage = stage
        self.llm_client = llm_client
        self.vectorstore = None
        self.threshold = 1
        
        if Config and VectorStore:
            # Patch Config to match actual directory structure on disk
            # Verified via list_dir: 'pre-action', 'post_observation', 'pre-planning'
            Config.DB_PATH_PRE_PLANNING = os.path.join(Config.BASE_DB_DIRECTORY, "pre-planning")
            Config.DB_PATH_PRE_ACTION = os.path.join(Config.BASE_DB_DIRECTORY, "pre-action")
            Config.DB_PATH_POST_OBSERVATION = os.path.join(Config.BASE_DB_DIRECTORY, "post_observation")
            
            # Select DB Type based on stage
            self.db_type = None
            if stage == 1:
                self.db_type = "pre_planning"
            elif stage == 2:
                self.db_type = "pre_action"
            elif stage == 3:
                self.db_type = "post_observation"
            elif stage == 4:
                self.db_type = "retrieval_phase"
            
            if self.db_type:
                print(f"Initializing Sandbox VectorStore for Stage {stage} ({self.db_type})...")
                self.vectorstore = VectorStore(db_type=self.db_type)
            else:
                print(f"VectorStore not initialized for Stage {stage}")

    def audit(self, content: str) -> Dict[str, Any]:
        """
        Main entry point for sandbox audit.
        
        Returns:
            Dict containing:
            - status: "SAFE", "UNSAFE", "RISK_DETECTED"
            - risk_description: Str (if detected)
            - content_to_agent: Str (sanitized or feedback)
            - analysis: Str (judge reasoning)
        """
        # 1. Extraction
        pattern, sanitized_content = self._extract_pattern(content)
        
        if not self.vectorstore or not pattern:
            # Fallback if DB not available or extraction failed
            return {
                "status": "SAFE",
                "risk_description": None,
                "content_to_agent": sanitized_content or content,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

        # 2. Vector Matching
        results = self.vectorstore.similarity_search(pattern, k=1)
        
        # Default Safe Response
        feedback = {
            "status": "SAFE",
            "risk_description": None,
            "content_to_agent": sanitized_content,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

        if results:
            top_doc, top_score = results[0]
            print(f"\n[SANDBOX AUDIT] Stage {self.stage} ({self.db_type or 'Unknown'}) | Similarity Score: {top_score:.4f}")
            print(f"Matched Case: {top_doc.page_content}")
            
            # 3. High Similarity Strategy (Score <= Threshold) - Cosine Distance
            if top_score <= self.threshold:
                risk_desc = top_doc.metadata.get("prompt", "Known Attack Pattern")
                is_harmful_meta = top_doc.metadata.get("is_harmful", True) # Default to harmful if not specified
                harm_label = "Malicious" if is_harmful_meta else "Benign" # Translating to Chinese as requested
                
                feedback["status"] = "RISK_DETECTED"
                # "大概率 [有害/无害]，这操作跟 [已知攻击描述] 类似"
                feedback["risk_description"] = f"Highly likely to be {harm_label}，This operation is similar to {top_doc.page_content}"
                feedback["content_to_agent"] = f"Alert: {feedback['risk_description']}.It is recommended to stop or modify the operation."
                feedback["matched_pattern"] = top_doc.page_content
                feedback["score"] = top_score
            
            # 4. Low Similarity Strategy (Score > Threshold)
            else:
                # Retrieve Top-K for context
                top_k_results = self.vectorstore.similarity_search(pattern, k=5) # User EDDF uses 5 refs
                feedback = self._run_online_judge(pattern, top_k_results, sanitized_content, content)

        return feedback

    def _extract_pattern(self, content: str) -> Tuple[str, str]:
        """
        Extracts abstract pattern from content using regex (finding content inside {Abstract Content: ...}).
        Returns:
            pattern: The abstract content to be vectorized.
            sanitized_content: The original content (used for feedback).
        """
        # Regex to find content inside {Abstract Content: ...}
        # This matches the latest Defense V2 output format
        match = re.search(r"\{Abstract Content:\s*(.*?)\}", content, re.DOTALL)
        
        if match:
            abstract_pattern = match.group(1).strip()
            return abstract_pattern, content
        
        # Fallback if specific tag not found
        return content, content

    def _run_online_judge(self, pattern, references, original_content, input_content) -> Dict[str, Any]:
        """
        Run LLM Judge with retrieved context (Low Similarity Strategy).
        Uses stage-specific templates (1-4).
        """
        # 1. Format Reference Cases
        reference_text_parts = []
        for i, (doc, score) in enumerate(references):
            # Extract 'pattern' or 'description' from metadata or page_content
            ref_pattern = doc.page_content
            reference_text_parts.append(f"Pattern (Ref {i+1}): {ref_pattern}")
        
        reference_block = "\n".join(reference_text_parts)
        
        # [USER REQUEST] Output detailed text for Top-k patterns (Low Similarity Scenario)
        print("-" * 80)
        print(f"[SANDBOX LOW SIMILARITY] Triggering Online Judge. Comparing Extracted Pattern vs Top-K References:")
        print(reference_block)
        print("-" * 80)
        
        # 2. Load Stage-Specific Template
        # current_file_path and workflow_root are defined at the top of the file
        template_dir = os.path.join(workflow_root, "template")
        
        # [MODIFIED] Map stages to actual filenames
        stage_map = {
            1: "sandbox_judge_query_stage.txt",
            2: "sandbox_judge_action_stage.txt", 
            3: "sandbox_judge_obs_stage.txt",
            4: "sandbox_judge_plan_stage.txt"
        }
        filename = stage_map.get(self.stage, f"sandbox_judge_stage_{self.stage}.txt")
        template_path = os.path.join(template_dir, filename)

        try:
            with open(template_path, "r") as f:
                template = f.read()
            # print(f"[DEBUG] Loaded template from {template_path}")
        except FileNotFoundError:
            # Fallback to generic or Stage 1 if specific not found
            print(f"[SANDBOX WARNING] Template {template_path} not found. Using stage 1 fallback.")
            try:
                 fallback_path = os.path.join(template_dir, "sandbox_judge_query_stage.txt") # Updated fallback name
                 with open(fallback_path, "r") as f:
                    template = f.read()
            except:
                 template = "Compare {CURRENT_PATTERN} with {REFERENCE_CASES}. Is {INPUT_CONTENT} harmful? Return JSON with boolean is_harmful."

        # 3. Fill Template
        prompt = template.replace("{{INPUT_CONTENT}}", input_content)
        prompt = prompt.replace("{{CURRENT_PATTERN}}", pattern)
        prompt = prompt.replace("{{REFERENCE_CASES}}", reference_block)
        
        # 4. Call LLM (GPT-4o)
        if self.llm_client:
            try:
                # Ensure we are asking for gpt-4o results
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o", 
                    messages=[
                        {"role": "system", "content": "You are a professional security evaluation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                     response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
                import json
                result_json = json.loads(result_text)
                
                is_harmful = result_json.get("is_harmful", False)
                reasoning = result_json.get("reasoning", "No reasoning provided.")
                true_problem = result_json.get("true_problem", "")

                status = "RISK_DETECTED" if is_harmful else "SAFE"
                
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                if response.usage:
                    token_usage["prompt_tokens"] = response.usage.prompt_tokens
                    token_usage["completion_tokens"] = response.usage.completion_tokens

                return {
                    "status": status,
                    "risk_description": f"By comparing and analyzing (Stage {self.stage})，Probably {'Harmful' if is_harmful else 'Benign'}。{reasoning}",
                    "content_to_agent": f"Alert: Potential risk detected.Analysis: {reasoning}" if is_harmful else original_content,
                    "analysis": reasoning,
                    "true_problem": true_problem,
                    "token_usage": token_usage
                }
                
            except Exception as e:
                print(f"[Sandbox Judge Error] {e}")
                return {
                     "status": "SAFE",
                     "risk_description": f"Judge execution failed: {e}",
                     "content_to_agent": original_content,
                     "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}
                }
        else:
             return {
                "status": "SAFE",
                "risk_description": "LLM Client not available for judgment.",
                "content_to_agent": original_content,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }
