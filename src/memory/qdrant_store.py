import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "true")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

SESSION_COLLECTION = "session_chunks"
MEMORY_COLLECTION = "verified_memory"


class ResearchVectorStore:
    def __init__(self, path: str = QDRANT_PATH):
        self.client = QdrantClient(path=path)
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        """Create Qdrant collections if they do not already exist."""
        existing = {c.name for c in self.client.get_collections().collections}

        for name in (SESSION_COLLECTION, MEMORY_COLLECTION):
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized dense embeddings locally on CPU."""
        return [vec.tolist() for vec in self.embedder.embed(texts)]

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        """Split clean article text into overlapping character chunks."""
        clean = " ".join(text.split())
        if not clean:
            return []
        chunks = []
        start = 0
        while start < len(clean):
            end = start + chunk_size
            chunks.append(clean[start:end])
            start += chunk_size - overlap
        return chunks

    def index_session_document(
        self,
        session_id: str,
        source_url: str,
        title: str,
        raw_text: str,
    ) -> int:
        """Chunk a scraped webpage, embed locally, and store in session_chunks."""
        chunks = self.chunk_text(raw_text)
        if not chunks:
            return 0

        vectors = self._embed(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "session_id": session_id,
                    "source_url": source_url,
                    "title": title,
                    "chunk_index": idx,
                    "text": chunk,
                },
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        self.client.upsert(collection_name=SESSION_COLLECTION, points=points)
        return len(points)

    def search_session_evidence(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks strictly scoped to the active session_id."""
        query_vector = self._embed([query])[0]

        results = self.client.query_points(
            collection_name=SESSION_COLLECTION,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id),
                    )
                ]
            ),
            limit=top_k,
        ).points

        return [
            {
                "score": round(hit.score, 4),
                "source_url": hit.payload.get("source_url"),
                "title": hit.payload.get("title"),
                "text": hit.payload.get("text"),
            }
            for hit in results
        ]

    def save_verified_memory(
        self,
        topic: str,
        summary: str,
        source_urls: List[str],
    ) -> str:
        """Persist Critic-approved research summary into long-term memory."""
        point_id = str(uuid.uuid4())
        vector = self._embed([f"{topic}: {summary}"])[0]

        self.client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "topic": topic,
                        "summary": summary,
                        "source_urls": source_urls,
                        "verified_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )
        return point_id

    def recall_verified_memory(
        self,
        query: str,
        score_threshold: float = 0.65,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Check if prior verified research already exists for a given topic."""
        query_vector = self._embed([query])[0]

        results = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            score_threshold=score_threshold,
            limit=top_k,
        ).points

        return [
            {
                "score": round(hit.score, 4),
                "topic": hit.payload.get("topic"),
                "summary": hit.payload.get("summary"),
                "source_urls": hit.payload.get("source_urls", []),
                "verified_at": hit.payload.get("verified_at"),
            }
            for hit in results
        ]

    def close(self) -> None:
        """Cleanly close local Qdrant storage before exit."""
        self.client.close()


if __name__ == "__main__":
    store = ResearchVectorStore()
    try:
        test_session = "test-session-001"

        added = store.index_session_document(
            session_id=test_session,
            source_url="https://example.com",
            title="2026 Enterprise AI Trends",
            raw_text=(
                "Model Context Protocol (MCP) has emerged as the standard interface for "
                "connecting autonomous LLM agents to enterprise databases and external APIs. "
                "Meanwhile, hybrid vector search combined with reflection loops reduces "
                "hallucinations in multi-step research pipelines."
            ),
        )
        print(f"Indexed {added} session chunks.")

        hits = store.search_session_evidence(test_session, "What is MCP used for?")
        print("Session Search Hits:", hits)

        mem_id = store.save_verified_memory(
            topic="Model Context Protocol Adoption",
            summary="MCP standardizes tool discovery and execution between LLM agents and external data sources.",
            source_urls=["https://example.com"],
        )
        print(f"Saved Long-Term Memory ID: {mem_id}")

        recalled = store.recall_verified_memory("How do agents connect to tools?")
        print("Long-Term Recall Hits:", recalled)
    finally:
        store.close()
