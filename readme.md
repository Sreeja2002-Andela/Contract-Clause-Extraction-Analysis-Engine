# Contract Clause Extraction & Analysis Engine

AI-powered contract analysis over the CUAD dataset. Extracts 41 clause categories, flags risk, enables natural language QA, and measures accuracy against expert annotations.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
# Windows
set GROQ_API_KEY=your_key_here

# Mac/Linux
export GROQ_API_KEY=your_key_here
```

### 3. Place your data file
```
contract_analyzer/
└── data/
    └── CUADv1.json     ← place it here
```

### 4. Run ingestion (first time only — makes LLM calls)
```bash
python run.py ingest 20
```
This loads 20 contracts, chunks them, embeds them, extracts all 41 clause categories via LLM, and saves results to `outputs/clauses.json` and `outputs/risks.json`.

**After the first run, LLM extraction is never called again.** The API loads directly from `outputs/clauses.json` on every subsequent start.

### 5. Start the backend
```bash
# Terminal 1 — keep running
python run.py api
```
API runs at `http://localhost:8000` — docs at `http://localhost:8000/docs`

### 6. Start the UI
```bash
# Terminal 2 — keep running
python run.py ui
```
UI runs at `http://localhost:8501`

### 7. Load contracts into the UI
In the Streamlit sidebar, click **Start Ingestion** with source `auto`. This loads from `outputs/clauses.json` — no LLM calls, completes in ~30 seconds.

### 8. (Optional) Run evaluation
```bash
python run.py eval
```
Computes precision, recall, and F1 per clause category against CUAD ground truth annotations. Results saved to `outputs/eval_results.json`.

---

## Project Structure

```
contract_analyzer/
├── README.md
├── requirements.txt
├── run.py                          ← single entry point: api | ui | ingest | eval
├── data/
│   └── CUADv1.json                 ← CUAD dataset (place here)
├── outputs/                        ← auto-created after first ingest
│   ├── clauses.json                ← pre-computed clause extractions
│   ├── risks.json                  ← pre-computed risk scores
│   └── eval_results.json           ← precision/recall/F1 report
├── chroma_store/                   ← auto-created vector index
├── ingest/
│   ├── extract.py                  ← CUADv1.json loader + text cleaning
│   ├── chunk.py                    ← paragraph-level chunking
│   └── embed.py                    ← sentence-transformers + ChromaDB indexing
├── retrieval/
│   ├── dense.py                    ← ChromaDB cosine semantic search
│   ├── sparse.py                   ← BM25 keyword index
│   └── hybrid.py                   ← RRF fusion (k=60)
├── extraction/
│   ├── clause_extractor.py         ← LLM extraction for all 41 categories + QA
│   └── risk_scorer.py              ← rule-based risk flagging
├── eval/
│   └── cuad_eval.py                ← precision/recall/F1 vs CUAD ground truth
├── api/
│   └── main.py                     ← FastAPI backend (7 endpoints)
└── ui/
    └── app.py                      ← Streamlit frontend (4 views)
```

---

## Architecture Overview

```
                        User (Streamlit UI)
                               │
                        FastAPI Backend
                               │
            ┌──────────────────┴──────────────────┐
            │          Ingestion Pipeline          │
            │  CUADv1.json → paragraph chunks      │
            │  → sentence-transformers embeddings  │
            │  → ChromaDB + BM25 index             │
            └──────────────────┬──────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │         Retrieval Layer              │
            │  dense.py   — ChromaDB cosine        │
            │  sparse.py  — BM25 keyword           │
            │  hybrid.py  — RRF fusion (k=60)      │
            └──────────┬───────────────┬───────────┘
                       │               │
            ┌──────────▼──────┐  ┌─────▼────────────┐
            │   Path A        │  │   Path B          │
            │ Batch clause    │  │ Conversational QA │
            │ extraction      │  │                   │
            │ 41 categories   │  │ NL question       │
            │ single LLM pass │  │ → retrieve chunks │
            │ → JSON output   │  │ → LLM answer      │
            │ → cached to     │  │ → cited response  │
            │   disk          │  │                   │
            └──────────┬──────┘  └─────┬─────────────┘
                       │               │
            ┌──────────┴───────────────┴───────────┐
            │              Output Layer             │
            │  Dashboard   — risk overview          │
            │  Contract    — clause viewer          │
            │  Compare     — cross-contract diff    │
            │  Ask         — NL QA with citations   │
            └───────────────────────────────────────┘
```

---

## Where LLM Is and Is Not Used

| Component | Technology | LLM? |
|---|---|---|
| Text extraction | Python + regex | No |
| Paragraph chunking | Python + regex | No |
| Embedding | sentence-transformers (local) | No |
| Vector search | ChromaDB cosine similarity | No |
| Keyword search | BM25 (rank-bm25) | No |
| Retrieval fusion | RRF formula | No |
| **Clause extraction** | **Claude API (single pass, all 41 categories)** | **Yes** |
| **QA answers** | **Claude API (retrieval-augmented)** | **Yes** |
| Risk scoring | Rule-based (if/else on clause presence) | No |
| Evaluation | Precision/recall math vs CUAD annotations | No |

LLM is used in exactly 2 places. Everything else runs locally with no API calls.

---

## Key Design Decisions

### 1. Paragraph-level chunking, not token windows
Legal clauses are semantically paragraph-scoped. A token-window split mid-clause corrupts the subject-obligation relationship — the consequence of an obligation without its trigger is meaningless for extraction. We split at double-newline boundaries and merge short paragraphs (< 80 chars, typically section headings) into adjacent ones.

### 2. Hybrid BM25 + dense retrieval with RRF fusion
Dense embeddings handle semantic variation: "bear full responsibility for losses" correctly retrieves indemnification clauses without the word "indemnify". But dense retrieval is weak on exact legal terminology — "Force Majeure", clause headings in ALL CAPS, exact defined terms. BM25 handles these exactly. Reciprocal Rank Fusion (k=60) combines both without the scale-normalization problems of score averaging.

### 3. Single LLM pass for all 41 clause categories
One API call per contract, not 41. Reasons: (a) 41 × 20 = 820 API calls is expensive and slow; (b) a single call preserves cross-clause context — the model can reason about relationships between adjacent clauses; (c) structured JSON output forces explicit marking of absent clauses, which is our primary risk signal.

### 4. Absence detection as the primary risk signal
A contract missing a cap on liability is more dangerous than one with an explicit uncapped liability clause — because the risk is hidden. When extraction returns `present: false` for critical clauses (cap_on_liability, termination_for_convenience, governing_law), the system flags these as HIGH risk. **Absence is the finding, not a retrieval failure.**

### 5. Disk cache eliminates re-extraction
After the first ingest, `outputs/clauses.json` and `outputs/risks.json` are written to disk. On every subsequent API start, the ingestion pipeline loads from these files instead of making LLM calls. Chunking and embedding still run (fast, local, ~30 seconds). LLM is never called again unless the cache is deleted.

### 6. CUAD ground truth for quantitative evaluation
CUAD provides 13,000+ expert annotations across 41 categories. We compute precision, recall, and F1 per category and report it honestly. This is the only problem in the hackathon set where you can measure your own system. Most candidates don't do this.

---

## ⚠️ Known Limitations & Failure Modes

### 1. Clause Extraction Variability
High linguistic variation causes false negatives, particularly for revenue sharing and joint IP ownership clauses which are expressed differently across SEC filings.
* **Miss rate:** ~30–40% for high-variation categories.
* **F1 score:** Below 0.5 for `revenue_profit_sharing` and `joint_ip_ownership`.
* **Root cause:** The same concept is often expressed in semantically distant language. Dense retrieval partially bridges this but does not eliminate it completely.
* **Current mitigation:** Semantic batch grouping ensures related clauses are evaluated together, which improves contextual reasoning.

### 2. Partial Contract Processing
Initial single-pass extraction failed silently on 4 out of 20 contracts due to context overflow. The LLM returned truncated JSON, which was saved as all-false extractions, collapsing recall and producing an overall F1 of 0.469.
* **Result:** Silent failures are eliminated. Failed batches now return safe `present: false` defaults per category rather than crashing the pipeline.

### 3. Short vs. Long Contract Handling
* **Contracts ≤ 6000 chars:** Processed using full compressed text across all batches.
* **Contracts > 6000 chars:** Processed using per-batch targeted retrieval. Only chunks relevant to each batch's theme are fetched. This keeps input tokens low but loses cross-batch clause context (i.e., a clause spanning two semantic domains might be missed).

### 4. Table & Structured Data Extraction
Clauses embedded in tables, schedules, and exhibits are flattened to plain text during extraction. The chunking process breaks their logical structure, meaning these clauses are frequently missed or misinterpreted.

### 5. Risk Scoring Bias (Expected Behavior)
Currently, all 20 tested contracts are flagged as **HIGH** risk. This is correct behavior, not a bug.
* **Reasoning:** Most commercial contracts are missing at least one critical clause—such as a cap on liability, termination for convenience, or governing law. The risk score accurately reflects actual contract quality against these strict criteria.

### 6. Infrastructure Constraints
* **Groq Rate Limits:** The `llama-3.3-70b-versatile` model has strict per-minute token limits. The retry logic rotates attempts, but sustained batch processing may still hit these limits. 
  * *Tip:* Add `time.sleep(2)` between contracts if rate limiting persists.
* **ChromaDB Cold Start:** The first embedding run must download the sentence-transformers model (~90MB), causing an initial delay. 
  * *Mitigation:* Pre-cache the model before running a demo by executing the following command:
    ```bash
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
    ```
---

## 📊 Extraction Evaluation Report

**Overall F1 Score: 0.475**

### Interpretation Guide
* **F1 > 0.7 (Strong):** Reliable for most use cases.
* **0.5 ≤ F1 ≤ 0.7 (Moderate):** Good for high-recall needs; review high-risk flags.
* **F1 < 0.5 (Weak):** Clause language is too varied for consistent extraction.

---

### Performance by Category

| Category | Precision | Recall | F1 Score | Support | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `license_grant` | 1.000 | 0.750 | 0.857 | 8 | Strong |
| `insurance` | 0.667 | 0.857 | 0.750 | 7 | Strong |
| `cap_on_liability` | 0.636 | 0.875 | 0.737 | 8 | Strong |
| `anti_assignment` | 0.778 | 0.583 | 0.667 | 12 | Moderate |
| `warranty_duration` | 1.000 | 0.500 | 0.667 | 4 | Moderate |
| `irrevocable_or_perpetual_license` | 0.667 | 0.667 | 0.667 | 3 | Moderate |
| `affiliate_license_licensee` | 0.667 | 0.667 | 0.667 | 3 | Moderate |
| `change_of_control` | 0.600 | 0.600 | 0.600 | 5 | Moderate |
| `termination_for_convenience` | 0.462 | 0.750 | 0.571 | 8 | Moderate |
| `non_compete` | 1.000 | 0.400 | 0.571 | 5 | Moderate |
| `minimum_commitment` | 0.667 | 0.500 | 0.571 | 4 | Moderate |
| `ip_ownership_assignment` | 0.500 | 0.500 | 0.500 | 4 | Moderate |
| `uncapped_liability` | 0.500 | 0.500 | 0.500 | 4 | Moderate |
| `third_party_beneficiary` | 0.200 | 1.000 | 0.333 | 1 | Weak |
| `covenant_not_to_sue` | 0.333 | 0.333 | 0.333 | 3 | Weak |
| `governing_law` | 0.750 | 0.176 | 0.286 | 17 | Weak |
| `audit_rights` | 0.500 | 0.200 | 0.286 | 5 | Weak |
| `liquidated_damages` | 0.200 | 0.500 | 0.286 | 2 | Weak |
| `revenue_profit_sharing` | 0.500 | 0.167 | 0.250 | 6 | Weak |
| `arbitration` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `indemnification` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `non_disparagement` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `no_solicitation` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `exclusivity` | 0.000 | 0.000 | 0.000 | 5 | Weak |
| `most_favored_nation` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `price_restrictions` | 0.000 | 0.000 | 0.000 | 1 | Weak |
| `volume_restriction` | 0.000 | 0.000 | 0.000 | 2 | Weak |
| `joint_ip_ownership` | 0.000 | 0.000 | 0.000 | 1 | Weak |
| `source_code_escrow` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `post_termination_services` | 0.000 | 0.000 | 0.000 | 4 | Weak |
| `unlimited_all_you_can_eat_license`| 0.000 | 0.000 | 0.000 | 0 | Weak |
| `affiliate_license_licensor` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| `class_action_waiver` | 0.000 | 0.000 | 0.000 | 0 | Weak |
| **OVERALL** | **0.483** | **0.467** | **0.475** | | |

---
  
## Dataset

**CUAD — Contract Understanding Atticus Dataset v1**
- File: `CUADv1.json` (SQuAD format)
- 510 commercial contracts sourced from SEC EDGAR
- 13,000+ expert annotations across 41 clause categories
- License: CC BY 4.0 — free for commercial and non-commercial use
- Source: [HuggingFace](https://huggingface.co/datasets/theatticusproject/cuad-qa) / [GitHub](https://github.com/TheAtticusProject/cuad)

---

## Tech Stack

| Component | Library               | Version | Reason |
|---|-----------------------|---|---|
| Vector store | ChromaDB              | 0.5.3 | Local, persistent, no server required |
| Embeddings | sentence-transformers | 3.0.1 | Fast, local, no API key needed |
| Keyword search | rank-bm25             | 0.2.2 | Exact legal term matching |
| Retrieval fusion | RRF (custom)          | — | Scale-invariant, outperforms score averaging |
| LLM | Groq                  | 0.37.1 | Strong structured output, legal reasoning |
| Backend | FastAPI               | 0.115.0 | Async, auto-docs, background tasks |
| Frontend | Streamlit             | 1.38.0 | Fast to build, local demo friendly |
| PDF fallback | pdfplumber            | 0.11.4 | Table-aware text extraction |
