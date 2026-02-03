# embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Global Model Cache for Singleton Pattern
import threading
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding  # 防止除以零
    return embedding / norm

class BaseEmbedding(Embeddings):
    def embed_documents(self, texts):
        raise NotImplementedError

    def embed_query(self, query):
        raise NotImplementedError

# 对于Jina模型的封装
class JinaEmbedding(BaseEmbedding):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer("../../jina-embeddings-v3", trust_remote_code=True)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, task="retrieval.passage", prompt_name="retrieval.passage", convert_to_numpy=True)
        return [normalize_embedding(embedding).tolist() for embedding in embeddings]

    def embed_query(self, query):
        query_embedding = self.model.encode([query], task="retrieval.query", prompt_name="retrieval.query", convert_to_numpy=True)
        return normalize_embedding(query_embedding).tolist()
class BertEmbedding(BaseEmbedding):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

class MiniLMEmbedding(BertEmbedding):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return [normalize_embedding(embedding).tolist() for embedding in embeddings]
        # return [embedding.tolist() for embedding in embeddings]
    def embed_query(self, query):
        query_embedding = self.model.encode(query)
        return normalize_embedding(query_embedding).tolist()
        # return query_embedding.tolist()
class GteQwenEmbedding(BertEmbedding):
    def __init__(self):
        global _MODEL_CACHE
        with _CACHE_LOCK:
            if "gte_Qwen2-1.5B-instruct" not in _MODEL_CACHE:
                try:
                    from modelscope import snapshot_download
                except ImportError:
                    from modelscope.hub.snapshot_download import snapshot_download
                # Try to use a local path for model cache instead of hardcoded server path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cache_path = os.path.join(current_dir, "../models")
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    
                model_dir = snapshot_download('iic/gte_Qwen2-1.5B-instruct', cache_dir=cache_path)
                # Set trust_remote_code=False to use the native transformers implementation
                # This avoids compatibility issues (like DynamicCache) with newer transformers versions
                _MODEL_CACHE["gte_Qwen2-1.5B-instruct"] = SentenceTransformer(model_dir, trust_remote_code=False)

            self.model = _MODEL_CACHE["gte_Qwen2-1.5B-instruct"]
            # In case you want to reduce the maximum length:
            # self.model.max_seq_length = 8192
        # In case you want to reduce the maximum length:
        # self.model.max_seq_length = 8192
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        # return [normalize_embedding(embedding).tolist() for embedding in embeddings]
        return [embedding.tolist() for embedding in embeddings]
    def embed_query(self, query):
        query_embedding = self.model.encode(query,prompt_name="query")
        # return normalize_embedding(query_embedding).tolist()
        return query_embedding.tolist()

class BGEM3Embedding(BertEmbedding):
    def __init__(self):
        global _MODEL_CACHE
        with _CACHE_LOCK:
            if "bge-m3" not in _MODEL_CACHE:
                try:
                    from modelscope import snapshot_download
                except ImportError:
                    from modelscope.hub.snapshot_download import snapshot_download
                # Try to use a local path for model cache instead of hardcoded server path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                cache_path = os.path.join(current_dir, "../models")
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path, exist_ok=True)
                    
                model_dir = snapshot_download('BAAI/bge-m3', cache_dir=cache_path)
                # Set trust_remote_code=False to use the native transformers implementation
                # This avoids compatibility issues (like DynamicCache) with newer transformers versions
                _MODEL_CACHE["bge-m3"] = SentenceTransformer(model_dir, trust_remote_code=False)

            self.model = _MODEL_CACHE["bge-m3"]
            # In case you want to reduce the maximum length:
            # self.model.max_seq_length = 8192
        # In case you want to reduce the maximum length:
        # self.model.max_seq_length = 8192
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        # return [normalize_embedding(embedding).tolist() for embedding in embeddings]
        return [embedding.tolist() for embedding in embeddings]
    def embed_query(self, query):
        query_embedding = self.model.encode(query,prompt_name="query")
        # return normalize_embedding(query_embedding).tolist()
        return query_embedding.tolist()

def get_embedding_model(model_name):
    if model_name == 'jina-embeddings-v3':
        return JinaEmbedding(model_name)
    elif model_name == 'bert-base-nli-mean-tokens':
        return BertEmbedding(model_name)
    elif model_name == 'all-MiniLM-L6-v2':
        return MiniLMEmbedding()
    elif model_name == "gte_Qwen2-1.5B-instruct":
        return GteQwenEmbedding()
    elif model_name == "BAAI/bge-m3":
        return BGEM3Embedding()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
