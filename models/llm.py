# models/llm.py
import os
import sys
from langchain_groq import ChatGroq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import settings

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        groq_api_key = settings.get("groq_api_key")
        model_name = settings.get("groq_model_name")
        
        if not groq_api_key:
            return None

        groq_model = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
        )
        return groq_model
    except Exception as e:
        print(f"Failed to initialize Groq model: {str(e)}")
        return None