"""
sparse.py — BM25 keyword index for exact legal term matching.

Why BM25 alongside dense retrieval?
Dense embeddings are weak on exact legal terminology:
- "Force Majeure" may embed near unrelated concepts
- "CDR", "SRR" acronyms have no semantic neighbors in embedding space
- Exact clause labels ("INDEMNIFICATION", "GOVERNING LAW") appear verbatim

BM25 catches these exact matches. Combined with dense via RRF fusion,
we get the best of both worlds.

The index is built in-memory at startup from the stored chunks.
For a hackathon demo with 20-510 contracts, this is fast enough.
"""

import re
from rank_bm25 import BM25Okapi


class BM25Index:
    """
    Lightweight wrapper around rank_bm25.
    Tokenizes by lowercased words; preserves legal terms.
    """

    def __init__(self):
        self.corpus_chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from a list of chunk dicts."""
        self.corpus_chunks = chunks
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[sparse] BM25 index built: {len(chunks)} chunks")

    def search(
        self,
        query: str,
        contract_id: str = None,
        top_k: int = 10
    ) -> list[dict]:
        """
        BM25 search. Optionally filter by contract_id after scoring.
        Returns list of result dicts with bm25_score.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        scored = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        hits = []
        for idx, score in scored:
            if score <= 0:
                continue
            chunk = self.corpus_chunks[idx]

            if contract_id and chunk["contract_id"] != contract_id:
                continue

            hits.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "contract_id": chunk["contract_id"],
                    "filename": chunk.get("filename", ""),
                    "para_idx": chunk["para_idx"],
                    "page_estimate": chunk["page_estimate"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                },
                "bm25_score": round(float(score), 4),
                "source": "sparse"
            })

            if len(hits) >= top_k:
                break

        return hits

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize preserving legal terms and acronyms.
        Lowercase, split on non-alphanumeric except hyphens.
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
        return tokens


# Module-level singleton
_bm25_index = BM25Index()


def build_bm25(chunks: list[dict]) -> None:
    """Build the module-level BM25 index from chunks."""
    _bm25_index.build(chunks)


def sparse_search(
    query: str,
    contract_id: str = None,
    top_k: int = 10
) -> list[dict]:
    """BM25 search using the module-level index."""
    return _bm25_index.search(query, contract_id=contract_id, top_k=top_k)