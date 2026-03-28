"""
api/main.py — FastAPI backend.

Endpoints:
  POST /ingest          — Load and index contracts
  GET  /contracts       — List loaded contracts
  GET  /contract/{id}   — Get a single contract's clause extraction + risk
  POST /qa              — Answer a NL question about a contract
  GET  /compare         — Compare a clause across multiple contracts
  GET  /risk-summary    — Risk dashboard across all loaded contracts
  GET  /health          — Health check

State is held in-process for hackathon simplicity.
In production, persist clause results to SQLite or Postgres.
"""

import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from ingest.chunk import chunk_all_contracts
from ingest.embed import embed_and_index
from retrieval.sparse import build_bm25
from extraction.clause_extractor import extract_clauses, answer_question, CLAUSE_CATEGORIES
from extraction.risk_scorer import score_contract, score_all_contracts

app = FastAPI(
    title="Contract Clause Extraction & Analysis Engine",
    description="AI-powered contract analysis using CUAD dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-process state
_state = {
    "contracts": {},       # {contract_id: contract_dict}
    "all_chunks": [],      # flat list of all chunks
    "clauses": {},         # {contract_id: clause_extraction_result}
    "risks": {},           # {contract_id: risk_result}
    "ingestion_status": "idle",  # idle | running | done | error
    "ingestion_log": []
}

CHROMA_PATH = "./chroma_store"


# ── Request/Response Models ──────────────────────────────────────────────────

class IngestRequest(BaseModel):
    cuad_json: str = "./data/CUADv1.json"  # primary: local CUADv1.json
    source: str = "auto"                    # auto | json | local | huggingface
    local_dir: Optional[str] = "./data/contracts"
    max_contracts: int = 20


class QARequest(BaseModel):
    contract_id: str
    question: str


class CompareRequest(BaseModel):
    contract_ids: list[str]
    clause: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "contracts_loaded": len(_state["contracts"]),
        "ingestion_status": _state["ingestion_status"]
    }


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger contract ingestion pipeline.
    Runs in background so the endpoint returns immediately.
    Poll /health for status.
    """
    if _state["ingestion_status"] == "running":
        raise HTTPException(409, "Ingestion already running")

    background_tasks.add_task(_run_ingestion, req)
    return {"message": "Ingestion started", "status": "running"}

def _run_ingestion(req: IngestRequest):
    _state["ingestion_status"] = "running"
    _state["ingestion_log"] = []

    def log(msg):
        print(msg)
        _state["ingestion_log"].append(msg)

    try:
        from ingest.extract import smart_load, load_cuad_json, load_cuad_texts, load_cuad_from_huggingface
        from pathlib import Path

        clauses_path = Path("./outputs/clauses.json")
        risks_path = Path("./outputs/risks.json")
        disk_cache_available = clauses_path.exists() and risks_path.exists()

        log(f"[ingest] Loading contracts (source={req.source}, max={req.max_contracts})")

        if req.source == "auto":
            contracts = smart_load(
                cuad_json=req.cuad_json,
                local_dir=req.local_dir,
                max_contracts=req.max_contracts
            )
        elif req.source == "json":
            contracts = load_cuad_json(req.cuad_json, max_contracts=req.max_contracts)
        elif req.source == "local":
            contracts = load_cuad_texts(req.local_dir, max_contracts=req.max_contracts)
        else:
            contracts = load_cuad_from_huggingface(max_contracts=req.max_contracts)

        if not contracts:
            raise ValueError("No contracts loaded")

        for c in contracts:
            _state["contracts"][c["contract_id"]] = c

        log(f"[ingest] Chunking {len(contracts)} contracts")
        chunks = chunk_all_contracts(contracts)
        _state["all_chunks"] = chunks

        log(f"[ingest] Building BM25 index ({len(chunks)} chunks)")
        build_bm25(chunks)

        log(f"[ingest] Embedding and indexing to ChromaDB")
        embed_and_index(chunks, chroma_path=CHROMA_PATH)

        # ── Smart cache: load from disk if available, skip LLM extraction ──
        if disk_cache_available:
            log(f"[ingest] Found outputs/clauses.json — loading pre-computed results (no LLM calls)")
            all_clauses = json.loads(clauses_path.read_text())
            all_risks = json.loads(risks_path.read_text())

            # Only load entries that match the contracts we just indexed
            loaded_ids = set(_state["contracts"].keys())
            matched = 0
            for cid in loaded_ids:
                if cid in all_clauses:
                    _state["clauses"][cid] = all_clauses[cid]
                    _state["risks"][cid] = all_risks.get(cid, score_contract(all_clauses[cid]))
                    matched += 1
                else:
                    # Contract not in cache — extract it now
                    log(f"[ingest] Cache miss, extracting: {cid}")
                    clauses = extract_clauses(_state["contracts"][cid], chroma_path=CHROMA_PATH)
                    _state["clauses"][cid] = clauses
                    _state["risks"][cid] = score_contract(clauses)

            log(f"[ingest] Loaded {matched} contracts from cache, "
                f"{len(loaded_ids) - matched} extracted fresh")
        else:
            # No cache — run full LLM extraction
            log(f"[ingest] No cache found. Extracting clauses from {len(contracts)} contracts via LLM...")
            for contract in contracts:
                cid = contract["contract_id"]
                log(f"[ingest] Extracting: {cid}")
                clauses = extract_clauses(contract, chroma_path=CHROMA_PATH)
                _state["clauses"][cid] = clauses
                _state["risks"][cid] = score_contract(clauses)

            # Save to disk for next time
            os.makedirs("./outputs", exist_ok=True)
            clauses_path.write_text(json.dumps(_state["clauses"], indent=2))
            risks_path.write_text(json.dumps(_state["risks"], indent=2))
            log(f"[ingest] Results cached to outputs/ for future runs")

        _state["ingestion_status"] = "done"
        log(f"[ingest] Done. {len(_state['contracts'])} contracts ready.")

    except Exception as e:
        _state["ingestion_status"] = "error"
        _state["ingestion_log"].append(f"ERROR: {str(e)}")
        print(f"[ingest] Error: {e}")


@app.get("/contracts")
def list_contracts():
    """List all loaded contracts with their risk level."""
    result = []
    for cid, contract in _state["contracts"].items():
        risk = _state["risks"].get(cid, {})
        result.append({
            "contract_id": cid,
            "filename": contract.get("filename", ""),
            "text_length": len(contract.get("raw_text", "")),
            "overall_risk": risk.get("overall_risk", "UNKNOWN"),
            "risk_score": risk.get("risk_score", 0),
            "clauses_extracted": sum(
                1 for v in _state["clauses"].get(cid, {}).values()
                if v.get("present")
            )
        })
    return {"contracts": result, "total": len(result)}


@app.get("/contract/{contract_id}")
def get_contract(contract_id: str):
    """Get full clause extraction and risk analysis for a contract."""
    if contract_id not in _state["contracts"]:
        raise HTTPException(404, f"Contract '{contract_id}' not found")

    contract = _state["contracts"][contract_id]
    clauses = _state["clauses"].get(contract_id, {})
    risk = _state["risks"].get(contract_id, {})

    present_clauses = {k: v for k, v in clauses.items() if v.get("present")}
    absent_clauses = {k: v for k, v in clauses.items() if not v.get("present")}

    return {
        "contract_id": contract_id,
        "filename": contract.get("filename", ""),
        "text_preview": contract["raw_text"][:500] + "...",
        "total_clauses_found": len(present_clauses),
        "total_clauses_absent": len(absent_clauses),
        "present_clauses": present_clauses,
        "absent_clauses": list(absent_clauses.keys()),
        "risk_analysis": risk
    }


@app.post("/qa")
def qa(req: QARequest):
    """Answer a natural language question about a specific contract."""
    if req.contract_id not in _state["contracts"]:
        raise HTTPException(404, f"Contract '{req.contract_id}' not found")

    contract = _state["contracts"][req.contract_id]
    result = answer_question(req.question, contract, chroma_path=CHROMA_PATH)
    return {"contract_id": req.contract_id, "question": req.question, **result}


@app.get("/compare")
def compare_contracts(clause: str, contract_ids: str):
    """
    Compare a specific clause across multiple contracts.
    contract_ids: comma-separated list
    clause: one of the 41 CUAD clause keys
    """
    ids = [c.strip() for c in contract_ids.split(",")]

    if clause not in CLAUSE_CATEGORIES:
        raise HTTPException(400, f"Unknown clause '{clause}'. Valid: {CLAUSE_CATEGORIES[:5]}...")

    comparison = []
    for cid in ids:
        if cid not in _state["clauses"]:
            continue
        clause_data = _state["clauses"][cid].get(clause, {"present": False})
        comparison.append({
            "contract_id": cid,
            "present": clause_data.get("present", False),
            "text": clause_data.get("text"),
            "page": clause_data.get("page")
        })

    present_count = sum(1 for c in comparison if c["present"])
    return {
        "clause": clause,
        "contracts_compared": len(comparison),
        "present_in": present_count,
        "absent_in": len(comparison) - present_count,
        "comparison": comparison
    }


@app.get("/risk-summary")
def risk_summary():
    """Risk dashboard across all loaded contracts."""
    if not _state["risks"]:
        return {"message": "No contracts loaded yet", "contracts": []}

    summary = []
    for cid, risk in _state["risks"].items():
        summary.append({
            "contract_id": cid,
            "overall_risk": risk["overall_risk"],
            "risk_score": risk["risk_score"],
            "summary": risk["summary"],
            "high_count": risk["high_count"],
            "medium_count": risk["medium_count"],
            "low_count": risk["low_count"],
            "top_flags": [f for f in risk["flags"] if f["level"] == "HIGH"][:3]
        })

    summary.sort(key=lambda x: -x["risk_score"])

    high_risk_contracts = [s for s in summary if s["overall_risk"] == "HIGH"]
    return {
        "total_contracts": len(summary),
        "high_risk_count": len(high_risk_contracts),
        "contracts": summary
    }


@app.get("/ingest-status")
def ingest_status():
    return {
        "status": _state["ingestion_status"],
        "contracts_loaded": len(_state["contracts"]),
        "log": _state["ingestion_log"][-20:]
    }

@app.post("/load-from-disk")
def load_from_disk():
    """Load pre-computed clauses from outputs/ — instant, no LLM calls."""
    import json
    from pathlib import Path
    from ingest.extract import smart_load
    from ingest.chunk import chunk_all_contracts
    from ingest.embed import embed_and_index
    from retrieval.sparse import build_bm25

    clauses_path = Path("./outputs/clauses.json")
    risks_path = Path("./outputs/risks.json")

    if not clauses_path.exists():
        raise HTTPException(404, "Run 'python run.py ingest' first")

    contracts = smart_load(max_contracts=200)
    for c in contracts:
        _state["contracts"][c["contract_id"]] = c

    chunks = chunk_all_contracts(contracts)
    _state["all_chunks"] = chunks
    build_bm25(chunks)
    embed_and_index(chunks, chroma_path=CHROMA_PATH)

    _state["clauses"] = json.loads(clauses_path.read_text())
    _state["risks"] = json.loads(risks_path.read_text())
    _state["ingestion_status"] = "done"

    return {"message": f"Loaded {len(_state['contracts'])} contracts from disk instantly"}