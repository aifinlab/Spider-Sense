# config.py
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = ''
os.environ["OPENAI_API_BASE"] = ''
os.environ["OPENROUTER_API_KEY"] = ''
os.environ["GOOGLE_API_KEY"]=''
class Config:
    # LLM配置
    # Switch to OpenRouter
    OPENAI_API_KEY = os.environ["OPENROUTER_API_KEY"] 
    OPENAI_API_BASE = ""
    
    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    
    model_name = "gpt-4o"
    # PERSIST_DIRECTORY = "./chroma/MiniLM-L6-v2"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: core -> EDDF -> Root (where chroma dir is)
    # Wait, original structure: EDDF/config.py -> chroma is in ../chroma?
    # Original: current_dir = EDDF/EDDF, project_root = EDDF. chroma = EDDF/chroma.
    # New: current_dir = EDDF/EDDF/core. We need project_root = EDDF.
    # So dirname(dirname(current_dir)) -> EDDF/EDDF/core -> EDDF/EDDF -> EDDF.
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Database Paths Config
    BASE_DB_DIRECTORY = "./HAC_db/chroma/bge-m3"
    
    # Define specific paths for different stages
    DB_PATH_PLANNING = os.path.join(BASE_DB_DIRECTORY, "planning_1")       # 原始/规划前
    DB_PATH_PRE_PLANNING = os.path.join(BASE_DB_DIRECTORY, "pre-planning") # 规划前 (New)
    DB_PATH_PRE_ACTION = os.path.join(BASE_DB_DIRECTORY, "pre-action")   # 行动前 (Fixed: underscore -> hyphen)
    DB_PATH_POST_OBSERVATION = os.path.join(BASE_DB_DIRECTORY, "post_observation") # 观察后 (Fixed: hyphen -> underscore)
    DB_PATH_RETRIEVAL_PHASE = os.path.join(BASE_DB_DIRECTORY, "retrieve_phase") # 检索阶段 (Stage 4)
    
    # Default to Planning (Plan A) for backward compatibility
    PERSIST_DIRECTORY = DB_PATH_PLANNING
    # EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME="BAAI/bge-m3"

