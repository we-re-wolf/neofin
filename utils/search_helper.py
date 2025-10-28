# utils/search_helper.py
from langchain_community.tools.tavily_search import TavilySearchResults
from config.config import settings

def get_web_search_tool():
    """Initialize and return the Tavily web search tool."""
    try:
        tavily_api_key = settings.get("tavily_api_key")
        
        if not tavily_api_key:
            return None
            
        search_tool = TavilySearchResults(
            api_key=tavily_api_key,
            k=3 
        )
        return search_tool
    except Exception as e:
        print(f"Error initializing Tavily search tool: {e}")
        return None