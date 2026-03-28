"""
hybrid.py — Reciprocal Rank Fusion (RRF) of dense + sparse retrieval.

Why RRF over score averaging?
- Dense and sparse scores are on different scales (cosine vs TF-IDF)
- Normalizing and averaging amplifies noise
- RRF only uses rank positions, making it scale-invariant
- Empirically outperforms weighted averaging on retrieval benchmarks

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each ranker i
k=60 is standard; it dampens the impact of very high ranks.

This is the retrieval layer that gets called for both:
- QA mode (single contract, user question)
- Extraction mode (clause-category query over one contract)
"""

from retrieval.dense import dense_search
from retrieval.sparse import sparse_search


RRF_K = 60


def hybrid_search(
    query: str,
    contract_id: str = None,
    top_k: int = 8,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    chroma_path: str = "./chroma_store"
) -> list[dict]:
    """
    Hybrid retrieval: dense + sparse fused with RRF.

    Args:
        query: Natural language query or clause-category description
        contract_id: If set, restrict search to a single contract
        top_k: Number of final results to return
        dense_top_k: Candidates to fetch from dense index
        sparse_top_k: Candidates to fetch from BM25

    Returns:
        List of chunk dicts sorted by RRF score, highest first.
        Each dict includes text, metadata, rrf_score, and contributing scores.
    """
    dense_results = dense_search(
        query, contract_id=contract_id, top_k=dense_top_k, chroma_path=chroma_path
    )
    sparse_results = sparse_search(
        query, contract_id=contract_id, top_k=sparse_top_k
    )

    rrf_scores = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores.setdefault(cid, {"rrf_score": 0.0, "result": result, "dense_rank": None, "sparse_rank": None})
        rrf_scores[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        rrf_scores[cid]["dense_score"] = result.get("dense_score", 0)
        rrf_scores[cid]["dense_rank"] = rank + 1

    for rank, result in enumerate(sparse_results):
        cid = result["chunk_id"]
        rrf_scores.setdefault(cid, {"rrf_score": 0.0, "result": result, "dense_rank": None, "sparse_rank": None})
        rrf_scores[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        rrf_scores[cid]["bm25_score"] = result.get("bm25_score", 0)
        rrf_scores[cid]["sparse_rank"] = rank + 1

    merged = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    final = []
    for item in merged[:top_k]:
        result = item["result"].copy()
        result["rrf_score"] = round(item["rrf_score"], 6)
        result["dense_rank"] = item.get("dense_rank")
        result["sparse_rank"] = item.get("sparse_rank")
        result["source"] = "hybrid"
        final.append(result)

    return final