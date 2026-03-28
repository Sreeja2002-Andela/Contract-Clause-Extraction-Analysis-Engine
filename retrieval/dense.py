"""
dense.py — Semantic retrieval via ChromaDB + sentence-transformers.

Dense retrieval handles semantic variation:
"bear full responsibility for losses" → matches "indemnification"
"either party may terminate without cause" → matches "termination for convenience"

This is what keyword search cannot do, and why this layer exists.
"""

from sentence_transformers import SentenceTransformer
from ingest.embed import get_chroma_collection, EMBED_MODEL_NAME

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def dense_search(
    query: str,
    contract_id: str = None,
    top_k: int = 10,
    chroma_path: str = "./chroma_store"
) -> list[dict]:
    """
    Semantic search over all indexed chunks.
    If contract_id is provided, filters to that contract only.

    Returns list of result dicts with text, metadata, and score.
    """
    model = get_model()
    _, collection = get_chroma_collection(chroma_path)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    where_filter = {"contract_id": contract_id} if contract_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    if not results["ids"] or not results["ids"][0]:
        return hits

    for i, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = 1 - distance  # cosine distance → similarity

        hits.append({
            "chunk_id": chunk_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_score": round(score, 4),
            "source": "dense"
        })

    return hits