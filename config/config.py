# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_config():
    """
    Loads all required API keys and model settings from environment variables.
    """
    try:
        config_settings = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY"),
            "groq_api_key": os.environ.get("GROQ_API_KEY"),
            "tavily_api_key": os.environ.get("TAVILY_API_KEY"),
            "google_api_key": os.environ.get("GOOGLE_API_KEY"), 
            "groq_model_name": os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
            "embedding_model_name": os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        }
        
        if not config_settings["groq_api_key"]:
            print("Warning: GROQ_API_KEY not found in environment variables.")
        if not config_settings["tavily_api_key"]:
            print("Warning: TAVILY_API_KEY not found in environment variables (required for web search).")
        # We no longer need a special check for embedding API keys   
        return config_settings

    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}

settings = get_config()