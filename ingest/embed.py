"""
embed.py — Embed chunks and store in ChromaDB.

Model choice: all-MiniLM-L6-v2
- 384-dim, fast, runs locally with no API key
- Strong on sentence-level semantic similarity
- Good enough for legal clause matching; swap for a larger model if needed

ChromaDB: local persistent store, no server required.
All-in-process for hackathon demo simplicity.
"""

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "contracts"
BATCH_SIZE = 64


def get_chroma_collection(chroma_path: str = CHROMA_PATH):
    """Return (or create) the ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def embed_and_index(chunks: list[dict], chroma_path: str = CHROMA_PATH) -> None:
    """
    Embed all chunks and upsert into ChromaDB.
    Idempotent: safe to re-run; existing chunk_ids are overwritten.
    """
    print(f"[embed] Loading model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    _, collection = get_chroma_collection(chroma_path)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "contract_id": c["contract_id"],
            "filename": c.get("filename", ""),
            "para_idx": c["para_idx"],
            "page_estimate": c["page_estimate"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
        }
        for c in chunks
    ]

    print(f"[embed] Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]

        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()

        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta
        )

    print(f"[embed] Indexed {len(chunks)} chunks into ChromaDB at {chroma_path}")


def get_embed_model() -> SentenceTransformer:
    """Singleton-style loader for the embedding model."""
    return SentenceTransformer(EMBED_MODEL_NAME)