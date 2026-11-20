import asyncio
import warnings
warnings.filterwarnings("ignore", message="coroutine 'ClientResponse.json' was never awaited")

import atexit
from typing import List, Dict, Any
from urllib.parse import urlparse

import trafilatura
import httpx
from ddgs import DDGS
from mcp.server.mcpserver import MCPServer

from src.memory.qdrant_store import ResearchVectorStore

# Initialize MCP v2 Server and single shared Qdrant Vector Store
mcp = MCPServer("EnterpriseResearchTools")
vector_store = ResearchVectorStore()
atexit.register(vector_store.close)

# Track scraped URLs per session in memory to avoid duplicate downloads
_scraped_urls_by_session: Dict[str, set[str]] = {}
_scrape_lock = asyncio.Lock()
_qdrant_lock = asyncio.Lock()



@mcp.tool()
def recall_long_term_memory(topic: str) -> List[Dict[str, Any]]:
    """
    Search persistent Qdrant memory for previously verified research summaries
    on a related topic before performing new web searches.
    """
    return vector_store.recall_verified_memory(query=topic, score_threshold=0.65, top_k=3)


@mcp.tool()
def search_web(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """
    Search the live web using DDGS, with an automatic Wikipedia API fallback
    so agents always receive reliable URLs and summaries.
    """
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results + 2):
                href = item.get("href", "")
                # Filter out sponsored ad redirects
                if not href or "://bing.com" in href or "googleadservices.com" in href:
                    continue
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": href,
                        "snippet": item.get("body", ""),
                    }
                )
                if len(results) >= max_results:
                    break
    except Exception:
        pass


    # Fallback to Wikipedia OpenSearch/Query API if DDGS returns empty
    if not results:
        try:
            resp = httpx.get(
                "https://wikipedia.org",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": max_results,
                },
                headers={"User-Agent": "MultiAgentResearchEngine/1.0 (https://github.com; research@example.com)"},

                timeout=10.0,
            )
            for item in resp.json().get("query", {}).get("search", []):
                title = item.get("title", "")
                slug = title.replace(" ", "_")
                snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
                results.append(
                    {
                        "title": title,
                        "url": f"https://wikipedia.org{slug}",
                        "snippet": snippet,
                    }
                )
        except Exception as exc:
            results.append({"title": "Search error", "url": "", "snippet": str(exc)})

    return results




@mcp.tool()
async def scrape_and_index_url(url: str, session_id: str) -> Dict[str, Any]:
    """
    Download a webpage, extract clean main article text via Trafilatura,
    chunk it, and index it into Qdrant under the active session_id.
    """
    async with _scrape_lock:
        seen = _scraped_urls_by_session.setdefault(session_id, set())
        if url in seen:
            return {"url": url, "status": "skipped_duplicate", "chunks_indexed": 0}
        seen.add(url)

    def _fetch_and_extract() -> tuple[str, str]:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "", ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True) or ""
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else urlparse(url).netloc
        return title, text

    title, clean_text = await asyncio.to_thread(_fetch_and_extract)

    if len(clean_text) < 150:
        return {"url": url, "status": "insufficient_content", "chunks_indexed": 0}

    async with _qdrant_lock:
        chunks_added = vector_store.index_session_document(
            session_id=session_id,
            source_url=url,
            title=title,
            raw_text=clean_text,
        )




    return {
        "url": url,
        "title": title,
        "status": "indexed",
        "chunks_indexed": chunks_added,
    }


@mcp.tool()
def retrieve_evidence(query: str, session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant grounded text chunks from Qdrant for a specific
    research sub-question within the active session_id.
    """
    return vector_store.search_session_evidence(
        session_id=session_id,
        query=query,
        top_k=top_k,
    )


@mcp.tool()
def save_to_long_term_memory(topic: str, summary: str, source_urls: List[str]) -> str:
    """
    Persist Critic-approved research summary into long-term Qdrant memory.
    """
    return vector_store.save_verified_memory(
        topic=topic,
        summary=summary,
        source_urls=source_urls,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
