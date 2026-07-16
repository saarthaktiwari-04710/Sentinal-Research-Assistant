from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the live web search engine
ddg_search = DuckDuckGoSearchRun()

@tool
def local_filesystem_search(query: str) -> str:
    """Searches the local vector database."""
    # Mock logic: If the query asks about our 'mocked' company data, return it.
    if "revenue" in query.lower() or "company" in query.lower():
        return f"[LOCAL DOC] Found highly relevant chunk for '{query}': 'Revenue grew by 14% due to AI integration.'"
    
    # Otherwise, return nothing so the CRAG Grader triggers the Web Search Fallback!
    return "No local documents found regarding this topic."

@tool
def web_search_fallback(query: str) -> str:
    """Searches the live web if local documents fail."""
    try:
        # Execute a real internet search
        search_results = ddg_search.invoke(query)
        return f"[WEB DOC] Live search results: {search_results}"
    except Exception as e:
        return f"Web search failed due to an error: {str(e)}"
