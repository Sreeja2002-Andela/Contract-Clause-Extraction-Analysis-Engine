"""
run.py — Start the full application.

Usage:
  python run.py api        # Start FastAPI backend on port 8000
  python run.py ui         # Start Streamlit UI on port 8501
  python run.py ingest     # Run ingestion pipeline directly (no UI)
  python run.py eval       # Run CUAD evaluation on loaded contracts
"""

import sys
import os
import json


def start_api():
    import uvicorn
    print("[run] Starting FastAPI backend at http://localhost:8000")
    print("[run] API docs at http://localhost:8000/docs")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


def start_ui():
    import subprocess
    print("[run] Starting Streamlit UI at http://localhost:8501")
    subprocess.run(["streamlit", "run", "ui/app.py", "--server.port=8501"])


def run_ingest(max_contracts: int = 20, source: str = "auto"):
    """Run ingestion directly — useful for pre-loading before demo."""
    from ingest.extract import smart_load, load_cuad_json, load_cuad_texts, load_cuad_from_huggingface
    from ingest.chunk import chunk_all_contracts
    from ingest.embed import embed_and_index
    from retrieval.sparse import build_bm25
    from extraction.clause_extractor import extract_clauses
    from extraction.risk_scorer import score_contract

    print(f"[run] Loading {max_contracts} contracts (source={source})")

    if source == "auto":
        contracts = smart_load(max_contracts=max_contracts)
    elif source == "json":
        contracts = load_cuad_json("./data/CUADv1.json", max_contracts=max_contracts)
    elif source == "local":
        contracts = load_cuad_texts("./data/contracts", max_contracts=max_contracts)
    else:
        contracts = load_cuad_from_huggingface(max_contracts=max_contracts)

    if not contracts:
        print("[run] No contracts loaded. Exiting.")
        return

    chunks = chunk_all_contracts(contracts)
    build_bm25(chunks)
    embed_and_index(chunks, chroma_path="./chroma_store")

    all_clauses = {}
    all_risks = {}
    for contract in contracts:
        cid = contract["contract_id"]
        print(f"[run] Extracting clauses: {cid}")
        clauses = extract_clauses(contract, chroma_path="./chroma_store")
        risk = score_contract(clauses)
        all_clauses[cid] = clauses
        all_risks[cid] = risk
        print(f"  → Risk: {risk['overall_risk']} | Found: {sum(1 for v in clauses.values() if v['present'])} clauses")

    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/clauses.json", "w") as f:
        json.dump(all_clauses, f, indent=2)
    with open("./outputs/risks.json", "w") as f:
        json.dump(all_risks, f, indent=2)

    print(f"\n[run] Done. Results saved to ./outputs/")
    print(f"[run] Total contracts: {len(contracts)}")
    print(f"[run] High risk: {sum(1 for r in all_risks.values() if r['overall_risk'] == 'HIGH')}")


def run_eval():
    """Run CUAD evaluation on previously extracted clauses."""
    import json
    from eval.cuad_eval import run_evaluation
    from ingest.extract import smart_load

    clauses_path = "./outputs/clauses.json"
    if not os.path.exists(clauses_path):
        print("[run] No extracted clauses found. Run: python run.py ingest first.")
        return

    with open(clauses_path) as f:
        all_clauses = json.load(f)

    # Reload contracts to get embedded ground truth from CUADv1.json
    contracts = smart_load(max_contracts=len(all_clauses))

    print(f"[run] Evaluating {len(all_clauses)} contracts against CUAD ground truth")
    metrics = run_evaluation(
        all_clauses,
        contracts=contracts,
        cuad_json_path="./data/CUADv1.json"
    )
    if metrics:
        overall = metrics.get("_overall", {})
        print(f"\nOverall F1: {overall.get('f1', 'N/A')}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "api"

    if cmd == "api":
        start_api()
    elif cmd == "ui":
        start_ui()
    elif cmd == "ingest":
        max_c = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        src = sys.argv[3] if len(sys.argv) > 3 else "auto"
        run_ingest(max_contracts=max_c, source=src)
    elif cmd == "eval":
        run_eval()
    else:
        print(__doc__)