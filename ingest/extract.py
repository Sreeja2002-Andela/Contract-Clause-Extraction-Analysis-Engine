"""
extract.py — Load contract text from CUADv1.json (primary path).

CUADv1.json structure (SQuAD format):
{
  "data": [
    {
      "title": "ContractName",
      "paragraphs": [
        {
          "context": "<full contract text>",
          "qas": [
            {
              "question": "...",
              "answers": [{"text": "...", "answer_start": N}],
              "id": "...",
              ...
            }
          ]
        }
      ]
    }
  ]
}

Each entry in "data" is one contract. "paragraphs" always has exactly one
item for CUAD — the full contract text is in paragraphs[0]["context"].
The "qas" list contains all 41 clause questions with their annotated answers.
We extract both the contract text AND the ground truth answers in one pass.
"""

import re
import json
import pdfplumber
from pathlib import Path
from typing import Optional


DEFAULT_CUAD_PATH = "./data/CUADv1.json"


def load_cuad_json(
    json_path: str = DEFAULT_CUAD_PATH,
    max_contracts: int = 20
) -> list[dict]:
    """
    Primary loader: reads CUADv1.json directly from disk.

    Returns list of contract dicts:
    {
        contract_id: str,
        filename: str,
        raw_text: str,
        source: "cuad_json",
        ground_truth: {question_text: [answer_texts]}  ← for eval
    }
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CUADv1.json not found at '{json_path}'.\n"
            f"Place CUADv1.json inside the data/ directory."
        )

    print(f"[extract] Loading CUADv1.json from {json_path} ...")
    with open(path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    data_entries = cuad.get("data", [])
    print(f"[extract] Found {len(data_entries)} contracts in CUADv1.json")

    contracts = []
    for entry in data_entries[:max_contracts]:
        title = entry.get("title", "unknown")
        paragraphs = entry.get("paragraphs", [])

        if not paragraphs:
            continue

        # CUAD always has exactly one paragraph per contract
        context = paragraphs[0].get("context", "")
        if len(context) < 500:
            continue

        # Extract ground truth answers for evaluation
        ground_truth = {}
        for qa in paragraphs[0].get("qas", []):
            question = qa.get("question", "")
            answers = qa.get("answers", [])
            answer_texts = [a["text"] for a in answers if a.get("text", "").strip()]
            ground_truth[question] = answer_texts

        contracts.append({
            "contract_id": sanitize_id(title),
            "filename": f"{title}.txt",
            "raw_text": clean_text(context),
            "source": "cuad_json",
            "ground_truth": ground_truth
        })

    print(f"[extract] Loaded {len(contracts)} contracts")
    return contracts


def sanitize_id(title: str) -> str:
    """Make contract title safe to use as a dictionary key / filename."""
    return re.sub(r"[^\w\-]", "_", title).strip("_")[:80]


def clean_text(text: str) -> str:
    """
    Clean contract text:
    - Remove null bytes
    - Normalize line endings
    - Collapse 3+ blank lines to 2
    - Strip common SEC filing header/footer noise
    """
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(
        r"^\s*(Page \d+ of \d+|CONFIDENTIAL|EXECUTION COPY)\s*$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    return text.strip()


def extract_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Fallback: extract text from a raw PDF using pdfplumber.
    Inserts [PAGE N] markers so chunk.py can estimate page numbers.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"[PAGE {i+1}]\n{text}")
    except Exception as e:
        print(f"[extract] PDF error {pdf_path}: {e}")
        return None
    return "\n\n".join(pages)


def load_cuad_texts(cuad_dir: str, max_contracts: int = 20) -> list[dict]:
    """
    Fallback: load from a directory of raw .txt or .pdf files.
    Used only if CUADv1.json is not available.
    """
    contracts = []
    cuad_path = Path(cuad_dir)
    files = sorted(cuad_path.glob("*.txt"))[:max_contracts] or \
            sorted(cuad_path.glob("*.pdf"))[:max_contracts]

    for f in files:
        try:
            if f.suffix == ".txt":
                text = f.read_text(encoding="utf-8", errors="replace")
            else:
                text = extract_from_pdf(str(f)) or ""
            text = clean_text(text)
            if len(text) > 500:
                contracts.append({
                    "contract_id": sanitize_id(f.stem),
                    "filename": f.name,
                    "raw_text": text,
                    "source": "local_file",
                    "ground_truth": {}
                })
        except Exception as e:
            print(f"[extract] Skipping {f.name}: {e}")

    print(f"[extract] Loaded {len(contracts)} contracts from {cuad_dir}")
    return contracts


def load_cuad_from_huggingface(max_contracts: int = 20) -> list[dict]:
    """
    Last-resort fallback: load CUAD from HuggingFace using parquet format.
    trust_remote_code is no longer supported — use parquet split directly.
    Only used if CUADv1.json is missing and no local files exist.
    """
    try:
        from datasets import load_dataset
        print("[extract] Falling back to HuggingFace CUAD (parquet)...")

        # Use the parquet-based version — no loading script required
        dataset = load_dataset(
            "theatticusproject/cuad-qa",
            split="train",
            data_files=None,
        )
        seen = {}
        for row in dataset:
            title = row.get("title", row.get("id", "unknown"))
            cid = sanitize_id(title)
            if cid not in seen:
                context = row.get("context", "")
                if len(context) > 500:
                    seen[cid] = {
                        "contract_id": cid,
                        "filename": f"{cid}.txt",
                        "raw_text": clean_text(context),
                        "source": "huggingface",
                        "ground_truth": {}
                    }
            if len(seen) >= max_contracts:
                break

        contracts = list(seen.values())
        print(f"[extract] Loaded {len(contracts)} contracts from HuggingFace")
        return contracts

    except Exception as e:
        print(f"[extract] HuggingFace fallback failed: {e}")
        print("[extract] Please place CUADv1.json in the data/ directory.")
        return []


def smart_load(
    cuad_json: str = DEFAULT_CUAD_PATH,
    local_dir: str = "./data/contracts",
    max_contracts: int = 20
) -> list[dict]:
    """
    Auto-selects the best available data source:
    1. CUADv1.json  (preferred — includes ground truth)
    2. Local .txt/.pdf directory
    3. HuggingFace stream (last resort)
    """
    if Path(cuad_json).exists():
        return load_cuad_json(cuad_json, max_contracts)
    if Path(local_dir).exists() and any(Path(local_dir).iterdir()):
        print(f"[extract] CUADv1.json not found, using local dir: {local_dir}")
        return load_cuad_texts(local_dir, max_contracts)
    print("[extract] No local data found, falling back to HuggingFace")
    return load_cuad_from_huggingface(max_contracts)