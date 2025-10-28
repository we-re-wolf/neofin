# models/embeddings.py
import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import settings

def get_openai_embeddings():
    """
    Initialize and return a local HuggingFace embedding model.
    (Note: Function name is kept for compatibility with app.py)
    """
    try:
    #     we are not using openai_api_key for now as it is not included in the free plan for docs
    #     openai_api_key = settings.get("openai_api_key") 
    #     if not openai_api_key:
    #         # Return None if key is not set, app.py will handle this
    #         return None
            
    #     embeddings = OpenAIEmbeddings(
    #         api_key=openai_api_key,
    #         model=model_name
    #     )
    #     return embeddings
    # except Exception as e:
    #     print(f"Failed to initialize OpenAI embeddings: {str(e)}")
    #     return None
        model_name = settings.get("embedding_model_name")
        
        if not model_name:
            print("Embedding model error: No embedding_model_name specified in config.")
            return None
            
        # Initialize HuggingFaceEmbeddings to run locally
        # This will download the model the first time it's run
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True} 
        )
        
        print(f"Successfully loaded local embedding model: {model_name}")
        return embeddings
        
    except Exception as e:
        print(f"Failed to initialize HuggingFace embeddings: {str(e)}")
        return None