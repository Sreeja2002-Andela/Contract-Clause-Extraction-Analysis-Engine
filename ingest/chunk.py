"""
chunk.py — Paragraph-level chunking for legal contracts.

Design decision: We chunk at paragraph boundaries, NOT arbitrary token windows.
Reason: Legal clauses are semantically paragraph-scoped. Splitting mid-clause
corrupts the subject-obligation relationship. A chunk containing only the
consequence of an obligation (without the trigger) is useless for extraction.

Each chunk carries full provenance metadata for citation.
"""

import re
from typing import Optional


def chunk_contract(contract: dict, max_chunk_chars: int = 1200) -> list[dict]:
    """
    Split a contract into paragraph-level chunks.

    Strategy:
    1. Split on double newlines (paragraph boundaries)
    2. Merge very short paragraphs (< 80 chars) into the next one
       — these are usually section headings, not standalone clauses
    3. Split very long paragraphs (> max_chunk_chars) at sentence boundaries
    4. Attach metadata: contract_id, page estimate, para_idx, char_start

    Returns list of chunk dicts ready for embedding.
    """
    raw_text = contract["raw_text"]
    contract_id = contract["contract_id"]

    raw_paragraphs = re.split(r"\n\s*\n", raw_text)

    merged = []
    buffer = ""
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) < 80 and not buffer:
            buffer = para
        elif buffer:
            merged.append(buffer + "\n" + para)
            buffer = ""
        else:
            merged.append(para)

    if buffer:
        merged.append(buffer)

    chunks = []
    para_idx = 0
    char_offset = 0

    for para in merged:
        if len(para) <= max_chunk_chars:
            page_estimate = estimate_page(char_offset, raw_text)
            chunks.append({
                "chunk_id": f"{contract_id}__p{para_idx}",
                "contract_id": contract_id,
                "filename": contract.get("filename", ""),
                "para_idx": para_idx,
                "page_estimate": page_estimate,
                "char_start": char_offset,
                "char_end": char_offset + len(para),
                "text": para,
                "text_length": len(para)
            })
            para_idx += 1
        else:
            sub_chunks = split_long_paragraph(para, max_chunk_chars)
            for sub in sub_chunks:
                page_estimate = estimate_page(char_offset, raw_text)
                chunks.append({
                    "chunk_id": f"{contract_id}__p{para_idx}",
                    "contract_id": contract_id,
                    "filename": contract.get("filename", ""),
                    "para_idx": para_idx,
                    "page_estimate": page_estimate,
                    "char_start": char_offset,
                    "char_end": char_offset + len(sub),
                    "text": sub,
                    "text_length": len(sub)
                })
                para_idx += 1
                char_offset += len(sub)
            continue

        char_offset += len(para)

    return chunks


def split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """
    Split a long paragraph at sentence boundaries.
    Fallback: hard split at max_chars if no sentence boundary found.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks


def estimate_page(char_offset: int, full_text: str) -> int:
    """
    Estimate page number from character offset.
    Heuristic: ~3000 chars per page (typical legal document).
    If [PAGE N] markers exist in the text (from PDF extraction), use those.
    """
    page_markers = [(m.start(), int(m.group(1)))
                    for m in re.finditer(r'\[PAGE (\d+)\]', full_text)]

    if page_markers:
        page = 1
        for marker_offset, marker_page in page_markers:
            if char_offset >= marker_offset:
                page = marker_page
            else:
                break
        return page

    return max(1, char_offset // 3000 + 1)


def chunk_all_contracts(contracts: list[dict], max_chunk_chars: int = 1200) -> list[dict]:
    """
    Chunk all contracts. Returns flat list of all chunks across all contracts.
    """
    all_chunks = []
    for contract in contracts:
        chunks = chunk_contract(contract, max_chunk_chars)
        all_chunks.extend(chunks)
        print(f"[chunk] {contract['contract_id']}: {len(chunks)} chunks")

    print(f"[chunk] Total: {len(all_chunks)} chunks from {len(contracts)} contracts")
    return all_chunks