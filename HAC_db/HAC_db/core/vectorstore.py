# vectorstore.py
import shutil
import os
from langchain_community.vectorstores import Chroma
from core.embedding import get_embedding_model  # Import the function to get the embedding model from embedding.py
from core.config import Config

class VectorStore:
    def __init__(self, db_type="planning"):
        """
        Constructor that selects the corresponding embedding model and database path.

        :param db_type: Type of database to use: "planning" (default) or "pre_action"
        """
        self.persist_directory = Config.DB_PATH_PLANNING  # Default
        
        if db_type == "pre_action":
            self.persist_directory = Config.DB_PATH_PRE_ACTION
        elif db_type == "post_observation":
            self.persist_directory = Config.DB_PATH_POST_OBSERVATION
        elif db_type == "pre_planning":
            self.persist_directory = Config.DB_PATH_PRE_PLANNING
        elif db_type == "planning":
            self.persist_directory = Config.DB_PATH_PLANNING
        elif db_type == "retrieval_phase":
            self.persist_directory = Config.DB_PATH_RETRIEVAL_PHASE
        
        # Ensure directory exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
            
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.embedding_function = get_embedding_model(self.model_name)  # Get the embedding model instance based on the model name
        # Determine Collection Name based on DB Type
        self.collection_name = "langchain" # Default
        if db_type == "pre_planning":
             self.collection_name = "pre_planning_patterns"
        elif db_type == "pre_action":
             self.collection_name = "pre_action_patterns" 
        elif db_type == "post_observation":
             self.collection_name = "post_observation_patterns"
        elif db_type == "retrieval_phase":
             self.collection_name = "retrieval_phase_patterns"
        
        # Initialize Chroma with explicit collection name
        self.vectorstore = Chroma(
                            persist_directory=self.persist_directory,
                            embedding_function=self.embedding_function,
                            collection_name=self.collection_name,
                            collection_metadata={"hnsw": "cosine"}
                            )

    def add_documents(self, texts, metadatas, ids=None, batch_size=10):
        """
        Add documents to the vector database.

        :param texts: List of text documents
        :param metadatas: List of corresponding metadata
        :param ids: Optional list of document IDs
        """
        if len(texts) != len(metadatas):
            raise ValueError("The lengths of the text list and metadata list do not match!")
        
        if ids and len(texts) != len(ids):
             raise ValueError("The lengths of the text list and ID list do not match!")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            batch_ids = ids[i:i + batch_size] if ids else None

            # Add the current batch of documents to the vector database
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)

            print(f"Successfully added {len(batch_texts)} documents to the database")
        print(f"Successfully added {len(texts)} documents to the database")

    def similarity_search(self, query, k=5):
        """
        Perform similarity search based on a query.

        :param query: The query text
        :param k: Number of most similar documents to return
        """
        return self.vectorstore.similarity_search_with_score(query=query, k=k)

    def clear_data(self):
        """
        Clear the current data in the vector database by deleting the database directory and its contents.
        """
        try:
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            print(f"Database directory '{self.persist_directory}' has been cleared")
        except Exception as e:
            print(f"An error occurred while clearing the database: {e}")


if __name__ == '__main__':
    import pandas as pd
    import json
    vectorstore = VectorStore()
    import os
    # Traverse all JSON files in the folder
    folder_path = ' '
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            dataset_path_test = os.path.join(folder_path, filename)
            with open(dataset_path_test, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            patterns = []
            prompts = []
            for item in train_data:
                # Check if the 'pattern' key exists and is not empty
                if 'pattern' in item and item['pattern']:
                    patterns.append(item['pattern'])
                    # Find the first value associated with a key containing "prompt"
                    prompt = next((value for key, value in item.items() if "prompt" in key), None)
                    # prompt=item["adversarial"]
                    prompts.append(prompt)
            # Create the metadatas list
            metadatas = [{"prompt": line} for line in prompts]
            # Clear the dataset if needed (commented out here)
            vectorstore.add_documents(patterns, metadatas)
            print(f"{dataset_path_test} completed")
