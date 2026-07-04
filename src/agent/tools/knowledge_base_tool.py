"""
src/agent/tools/knowledge_base_tool.py

Knowledge base RAG tool — searches embedded insurance documents
stored in ChromaDB for answers to general questions.

Use this tool for: coverage explanations, process guides, FAQ answers,
document requirements — anything that lives in written policy documents.

Do NOT use this tool for: live policy data, claim status, customer-
specific information. Those require the CRM tool (deterministic data
must come from authoritative systems, not from document search).

Architecture note:
    This tool performs DENSE retrieval only (embedding similarity).
    Hybrid retrieval (dense + BM25 keyword) can be added in a future
    version if recall on specific terms (policy numbers, dates) is poor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# The embedding model must match the model used when documents were indexed.
# Changing this model requires re-indexing the entire knowledge base.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Tool definition ───────────────────────────────────────────────────────────
# This dict is sent to the LLM so it knows this tool exists.
# The description is prompt engineering — it must tell the LLM exactly
# when to use this tool vs other tools.
TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the Meridian Insurance knowledge base for general information. "
            "Use this for: coverage explanations, claims process guides, FAQ answers, "
            "document requirements, billing questions, and general policy information. "
            "Do NOT use this for live policy data or claim status — use lookup_policy "
            "or get_claim_status instead. "
            "Returns the most relevant document excerpts with similarity scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The question or topic to search for. "
                        "Write this as a natural language question, "
                        "e.g. 'does home insurance cover water damage from burst pipes?'"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, maximum 10.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ── ChromaDB client ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """
    Get the ChromaDB collection, cached after first call.

    We cache this because:
    - Creating the client involves a network connection to ChromaDB
    - Loading the embedding model takes ~1 second on first call
    - The collection doesn't change between requests

    lru_cache(maxsize=1) means we keep exactly one cached connection.
    """
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    # The embedding function must match what was used during indexing
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},   # cosine similarity, not euclidean
    )

    log.info(
        "kb.collection_connected",
        collection=settings.chroma_collection_name,
        host=settings.chroma_host,
    )
    return collection


# ── Tool handler ──────────────────────────────────────────────────────────────

async def run(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Search the knowledge base for documents relevant to the query.

    Called by src/agent/tools/__init__.py dispatch() when the LLM
    requests the search_knowledge_base tool.

    Args:
        query: natural language question from the LLM
        top_k: number of results to return (capped at 10)

    Returns:
        dict with:
            found       (bool)   — True if at least one result above threshold
            best_score  (float)  — similarity score of top result (0.0 to 1.0)
            results     (list)   — list of document dicts with text and metadata
            error       (str)    — only present if something went wrong
    """
    top_k = min(top_k, 10)   # cap to prevent abuse

    try:
        collection = _get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            log.info("kb.no_results", query=query[:60])
            return {
                "found": False,
                "best_score": 0.0,
                "results": [],
            }

        # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite)
        # Convert to SIMILARITY (1=identical, 0=no match) for readability
        formatted = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            similarity = 1 - distance
            formatted.append({
                "doc_id": meta.get("doc_id", "unknown"),
                "title": meta.get("title", "Untitled"),
                "section": meta.get("section", ""),
                "text": doc,
                "similarity_score": round(similarity, 4),
            })

        best_score = formatted[0]["similarity_score"]

        log.info(
            "kb.search_complete",
            query=query[:60],
            results_count=len(formatted),
            best_score=best_score,
        )

        return {
            "found": best_score >= settings.agent_low_confidence_threshold,
            "best_score": best_score,
            "results": formatted,
        }

    except Exception as exc:
        log.error("kb.search_failed", query=query[:60], error=str(exc))
        return {
            "found": False,
            "best_score": 0.0,
            "results": [],
            "error": str(exc),
        }