"""
clause_extractor.py — Batched LLM extraction for all 41 CUAD clause categories.

Improvements over v1:
  1. ACCURACY   — Semantic batching (related clauses together), longer text quotes,
                  confidence field, richer system prompt with examples.
  2. JSON       — Smaller batches (10) to stay well under token limits, stricter
                  validation, per-field fallback instead of silently dropping keys.
  3. TOKEN COST — Contract text compressed once and reused across batches via a
                  shared context cache; retrieval mode used aggressively for long
                  contracts so we never repeat 8k-token inputs 3× per contract.
"""

import json
import os
import re
import groq
from dotenv import load_dotenv
from retrieval.hybrid import hybrid_search

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=api_key)

MODELS = [
    "llama-3.3-70b-versatile",
]

# ---------------------------------------------------------------------------
# Semantic batches — related clauses grouped together so the model has
# coherent context when deciding presence/absence.
# Smaller batches (≤10) keep JSON output well under 2048 tokens.
# ---------------------------------------------------------------------------
CLAUSE_CATEGORIES = [
    "document_name", "parties", "agreement_date", "effective_date",
    "expiration_date", "renewal_term", "notice_period_to_terminate_renewal",
    "governing_law", "most_favored_nation", "non_compete", "exclusivity",
    "no_solicitation", "non_disparagement", "termination_for_convenience",
    "change_of_control", "anti_assignment", "revenue_profit_sharing",
    "price_restrictions", "minimum_commitment", "volume_restriction",
    "ip_ownership_assignment", "joint_ip_ownership", "license_grant",
    "non_transferable_license", "affiliate_license_licensor",
    "affiliate_license_licensee", "unlimited_all_you_can_eat_license",
    "irrevocable_or_perpetual_license", "source_code_escrow",
    "post_termination_services", "audit_rights", "uncapped_liability",
    "cap_on_liability", "liquidated_damages", "warranty_duration",
    "insurance", "covenant_not_to_sue", "third_party_beneficiary",
    "class_action_waiver", "arbitration", "indemnification",
]

SEMANTIC_BATCHES = [
    # Batch 1 — Contract identity & timeline
    ["document_name", "parties", "agreement_date", "effective_date",
     "expiration_date", "renewal_term", "notice_period_to_terminate_renewal",
     "governing_law"],

    # Batch 2 — Competitive & relationship restrictions
    ["most_favored_nation", "non_compete", "exclusivity", "no_solicitation",
     "non_disparagement", "anti_assignment", "change_of_control",
     "termination_for_convenience"],

    # Batch 3 — Financial terms
    ["revenue_profit_sharing", "price_restrictions", "minimum_commitment",
     "volume_restriction", "cap_on_liability", "uncapped_liability",
     "liquidated_damages", "insurance"],

    # Batch 4 — IP & licensing
    ["ip_ownership_assignment", "joint_ip_ownership", "license_grant",
     "non_transferable_license", "affiliate_license_licensor",
     "affiliate_license_licensee", "unlimited_all_you_can_eat_license",
     "irrevocable_or_perpetual_license", "source_code_escrow"],

    # Batch 5 — Post-contract, dispute resolution & misc
    ["post_termination_services", "audit_rights", "warranty_duration",
     "covenant_not_to_sue", "third_party_beneficiary",
     "class_action_waiver", "arbitration", "indemnification"],
]

# Map every category to its batch index for targeted retrieval queries
BATCH_RETRIEVAL_QUERIES = {
    0: "agreement date parties effective date governing law renewal termination notice",
    1: "non-compete exclusivity no-solicitation assignment change of control termination convenience",
    2: "revenue sharing price minimum commitment liability cap liquidated damages insurance",
    3: "intellectual property ownership license grant affiliate perpetual irrevocable source code escrow",
    4: "post-termination audit warranty arbitration indemnification class action waiver beneficiary",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a senior legal analyst specialising in commercial contract review (CUAD benchmark).

OUTPUT FORMAT — strictly valid JSON, nothing else:
{
  "clause_key": {
    "present": true | false,
    "text": "verbatim quote from contract (up to 500 chars) or null",
    "page": estimated_page_integer_or_null,
    "confidence": "high" | "medium" | "low"
  }
}

RULES (follow exactly):
1. Output ONLY the JSON object. No preamble, no explanation, no markdown fences.
2. Start with { and end with }
3. Use ONLY the clause keys listed in the user message — no extras.
4. present=true  → text must be a non-null quote from the contract.
5. present=false → text must be null.
6. confidence="high"   when the clause is explicit and unambiguous.
   confidence="medium" when inferred or partially present.
   confidence="low"    when uncertain.
7. page: estimate from section headers / page markers in the text; null if unknown.
8. Quote the most legally significant sentence(s), not the entire paragraph.
"""

EXTRACTION_USER_TEMPLATE = """Contract excerpt:
\"\"\"
{contract_text}
\"\"\"

Extract EXACTLY these clause categories (no others):
{category_list}

Respond with a single JSON object starting with {{ and ending with }}.
Each key must map to: {{"present": bool, "text": string|null, "page": int|null, "confidence": "high"|"medium"|"low"}}"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_clauses(contract: dict, chroma_path: str = "./chroma_store") -> dict:
    """
    Extract all 41 CUAD clause categories from a contract.

    Token-reduction strategy:
    - Contracts ≤ 6000 chars → use raw text directly (one compress, reuse across batches).
    - Contracts > 6000 chars → per-batch targeted retrieval (fetch only relevant chunks).
      This avoids sending 8000 tokens × 5 batches = 40k input tokens per contract.

    Returns dict: {clause_category: {present, text, page, confidence}}
    """
    contract_id = contract["contract_id"]
    raw_text = contract["raw_text"]

    SHORT_CONTRACT_LIMIT = 6000  # chars ~ 1500 tokens; safe to include in every batch

    if len(raw_text) <= SHORT_CONTRACT_LIMIT:
        mode = "full"
        shared_text = _compress_text(raw_text, SHORT_CONTRACT_LIMIT)
        print(f"[extractor] {contract_id}: full-text mode ({len(shared_text)} chars)")
    else:
        mode = "retrieval"
        print(f"[extractor] {contract_id}: retrieval mode (contract too long)")

    final_result = {}

    for batch_idx, batch in enumerate(SEMANTIC_BATCHES):
        print(f"[extractor] Batch {batch_idx + 1}/{len(SEMANTIC_BATCHES)}: {batch[:3]}...")

        if mode == "full":
            contract_text = shared_text
        else:
            # Targeted retrieval per batch — only fetch chunks relevant to this batch's themes
            contract_text = _retrieve_for_batch(contract_id, batch_idx, chroma_path)

        category_list = ", ".join(f'"{c}"' for c in batch)
        user_message = EXTRACTION_USER_TEMPLATE.format(
            contract_text=contract_text,
            category_list=category_list,
        )

        batch_result = _call_llm_with_retry(contract_id, batch, user_message)
        final_result.update(batch_result)

    return validate_and_fill(final_result)


def answer_question(
    question: str,
    contract: dict,
    chroma_path: str = "./chroma_store"
) -> dict:
    """
    Answer a natural language question about a specific contract.
    Returns: {answer, sources, confidence}
    """
    contract_id = contract["contract_id"]

    hits = hybrid_search(question, contract_id=contract_id, top_k=5, chroma_path=chroma_path)

    if not hits:
        return {
            "answer": "I could not find relevant information in this contract.",
            "sources": [],
            "confidence": "low",
        }

    context_parts = []
    for i, hit in enumerate(hits):
        page = hit["metadata"].get("page_estimate", "?")
        context_parts.append(f"[Source {i + 1}, Page ~{page}]\n{hit['text']}")

    context = "\n\n".join(context_parts)

    system = (
        "You are a legal contract analyst. Answer questions about contracts accurately.\n"
        "Always cite the source number (e.g., [Source 1]) when referring to specific text.\n"
        "If the answer is not in the provided context, say so clearly — do not guess.\n"
        "Keep answers concise and precise. Where a clause exists, quote the key sentence."
    )

    user = (
        f"Contract: {contract_id}\n\n"
        f"RELEVANT CONTRACT SECTIONS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a direct answer with citations."
    )

    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1024,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()
            sources = [
                {
                    "text": h["text"][:200] + "..." if len(h["text"]) > 200 else h["text"],
                    "page": h["metadata"].get("page_estimate"),
                    "para_idx": h["metadata"].get("para_idx"),
                    "rrf_score": h.get("rrf_score", 0),
                }
                for h in hits
            ]
            return {"answer": answer, "sources": sources, "confidence": "high"}

        except groq.RateLimitError:
            print(f"[qa] Rate limit on {model}, trying next...")
            continue
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": [], "confidence": "error"}

    return {
        "answer": "All models rate-limited. Please try again later.",
        "sources": [],
        "confidence": "error",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_text(text: str, max_chars: int) -> str:
    """
    Reduce contract text for the LLM without losing clause-bearing sentences.

    Strategy (in order):
    1. Strip excessive blank lines (common in PDFs).
    2. Remove lines that are purely numeric (page numbers, exhibit numbers).
    3. Hard-truncate to max_chars if still too long.

    This is purely string manipulation — zero API calls.
    """
    # Collapse 3+ blank lines to 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove standalone page-number lines like "    4   " or "Page 4 of 52"
    text = re.sub(r"(?m)^\s*(Page\s+\d+\s+of\s+\d+|\d+)\s*$", "", text)
    text = text.strip()
    return text[:max_chars]


def _retrieve_for_batch(contract_id: str, batch_idx: int, chroma_path: str) -> str:
    """
    Retrieve only the chunks relevant to a given semantic batch.
    Keeps input tokens low for long contracts.
    """
    query = BATCH_RETRIEVAL_QUERIES.get(batch_idx, "contract clause terms conditions")
    hits = hybrid_search(query, contract_id=contract_id, top_k=4, chroma_path=chroma_path)

    parts = []
    seen = set()
    for hit in hits:
        cid = hit["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            page = hit["metadata"].get("page_estimate", "?")
            parts.append(f"[Page ~{page}]\n{hit['text']}")

    return "\n\n---\n\n".join(parts) if parts else "(No relevant sections found)"


def _call_llm_with_retry(contract_id: str, batch: list, user_message: str, retries: int = 3) -> dict:
    """
    Call LLM with retry + model rotation.  Returns a dict for the batch categories.
    On total failure, returns safe empty entries (present=False) for each category
    so downstream code never gets KeyError.
    """
    last_raw = ""

    for attempt in range(retries):
        model = MODELS[attempt % len(MODELS)]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=2048,
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()
            last_raw = raw
            cleaned = _clean_json(raw)

            if not cleaned:
                print(f"[extractor] Attempt {attempt + 1} ({model}): unparseable response")
                print(f"  Raw preview: {raw[:200]!r}")
                continue

            parsed = json.loads(cleaned)

            # Validate that at least half the expected keys are present
            found_keys = [k for k in batch if k in parsed]
            if len(found_keys) < len(batch) // 2:
                print(f"[extractor] Attempt {attempt + 1}: only {len(found_keys)}/{len(batch)} keys returned, retrying")
                continue

            return parsed

        except json.JSONDecodeError as e:
            print(f"[extractor] JSON error attempt {attempt + 1} ({model}): {e}")
            print(f"  Raw preview: {last_raw[:300]!r}")
        except groq.RateLimitError as e:
            print(f"[extractor] Rate limit on {model}: {e}")
        except Exception as e:
            print(f"[extractor] Unexpected error attempt {attempt + 1} ({model}): {e}")

    # All retries exhausted — return safe empty for each category in this batch
    print(f"[extractor] All retries failed for batch {batch[:2]}... in {contract_id}. Using empty defaults.")
    return {cat: {"present": False, "text": None, "page": None, "confidence": "low"} for cat in batch}


def _clean_json(raw: str) -> str:
    """
    Extract valid JSON from LLM output.  Handles in order:
      1. Markdown fences
      2. Preamble / postamble text
      3. Truncated JSON (trim to last complete entry)
      4. Unclosed braces (up to 3)
    Returns empty string if nothing recoverable.
    """
    if not raw:
        return ""

    # 1. Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # 2. Isolate the outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""

    json_str = raw[start: end + 1]

    # 3. Try as-is
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # 4. Find last COMPLETE key-value block.
    #    A complete block ends with either `}` followed by `,` or `}` followed by newline+`"`.
    #    We look for the last `},\n` or `}\n` pattern inside a nested object.
    patterns = [
        r'\}\s*,\s*\n\s*"',   # "},\n  "next_key"
        r'\}\s*\n\s*"',        # "}\n  "next_key" (no comma — shouldn't happen but defensive)
        r'\}\s*,',             # "},"  anywhere
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, json_str))
        if matches:
            cut = matches[-1].start() + 1  # keep the closing } of the last complete entry
            trimmed = json_str[:cut] + "\n}"
            try:
                json.loads(trimmed)
                print(f"[extractor] Recovered truncated JSON via pattern '{pat}'")
                return trimmed
            except json.JSONDecodeError:
                pass

    # 5. Close unclosed braces (up to 4)
    open_count = json_str.count("{") - json_str.count("}")
    if 0 < open_count <= 4:
        patched = json_str + ("}" * open_count)
        try:
            json.loads(patched)
            print(f"[extractor] Recovered JSON by appending {open_count} closing brace(s)")
            return patched
        except json.JSONDecodeError:
            pass

    return ""


def validate_and_fill(result: dict) -> dict:
    """
    Ensure all 41 categories are present.
    Normalise unexpected structures.
    Add confidence field if missing.
    """
    validated = {}
    for cat in CLAUSE_CATEGORIES:
        entry = result.get(cat)
        if isinstance(entry, dict):
            present = bool(entry.get("present", False))
            validated[cat] = {
                "present": present,
                "text": entry.get("text") if present else None,
                "page": entry.get("page"),
                "confidence": entry.get("confidence", "medium") if present else "low",
            }
        else:
            validated[cat] = {
                "present": False,
                "text": None,
                "page": None,
                "confidence": "low",
            }
    return validated


def get_empty_result() -> dict:
    return {
        cat: {"present": False, "text": None, "page": None, "confidence": "low"}
        for cat in CLAUSE_CATEGORIES
    }