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

## Known Limitations and Failure Modes

**False negatives on high-variation clauses**: Revenue sharing and joint IP ownership clauses vary too widely in language across SEC filings. The extractor misses roughly 30–40% of these. F1 drops below 0.5 for these categories. The fix would be more retrieval candidates before the LLM pass, or category-specific prompts.

**Long contracts exceed context window**: Contracts over ~60,000 characters fall back to retrieval-based extraction — top chunks per clause group are fetched and sent instead of the full text. This loses cross-clause coherence and increases miss rate.

**Table-formatted clauses not reliably extracted**: Schedules and exhibits formatted as tables (payment milestones, territory restrictions) are linearized poorly during text extraction and chunked incorrectly. These clauses are frequently missed.

**ChromaDB cold start on first embed**: The first embedding run downloads the sentence-transformers model (~90MB). Subsequent runs use the cached model. Ensure internet access before demo or pre-cache the model with:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**All 20 contracts flagged HIGH risk**: This is expected, not a bug. Most commercial contracts are missing at least one of the three critical clauses we check (cap on liability, termination for convenience, governing law). The risk score reflects the actual state of the contracts.

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
| LLM | Groq                  | latest | Strong structured output, legal reasoning |
| Backend | FastAPI               | 0.115.0 | Async, auto-docs, background tasks |
| Frontend | Streamlit             | 1.38.0 | Fast to build, local demo friendly |
| PDF fallback | pdfplumber            | 0.11.4 | Table-aware text extraction |