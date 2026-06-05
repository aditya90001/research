from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search web using Tavily"""
    results = tavily.search(query=query, max_results=5, search_depth="advanced")

    output = []
    for r in results.get("results", []):
        output.append(
            f"Title: {r.get('title')}\n"
            f"URL: {r.get('url')}\n"
            f"Snippet: {r.get('content')[:300]}\n"
        )

    return "\n---\n".join(output)


@tool
def scrape_url(url: str) -> str:
    """Scrape a webpage"""
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    return soup.get_text(" ", strip=True)[:5000]