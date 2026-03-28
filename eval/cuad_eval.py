"""
cuad_eval.py — Evaluate extraction accuracy against CUAD ground truth annotations.

This is our key differentiator: we can MEASURE ourselves.
CUAD provides expert-annotated spans for all 41 clause categories.

Metrics:
- Precision: of clauses we marked present, how many actually are? (TP / (TP + FP))
- Recall: of clauses that exist, how many did we find? (TP / (TP + FN))
- F1: harmonic mean of precision and recall

We evaluate at the presence/absence level (binary), not span-exact-match,
because span-exact matching is very strict and our extraction finds paraphrases.

Interpretation guide:
- F1 > 0.7: Strong. Reliable for most use cases.
- F1 0.5-0.7: Moderate. Good for high-recall needs; review high-risk flags.
- F1 < 0.5: Weak. Clause language too varied for this category.
"""

import json
from collections import defaultdict
from datasets import load_dataset


CUAD_CATEGORY_MAP = {
    "Governing Law": "governing_law",
    "Termination For Convenience": "termination_for_convenience",
    "Cap On Liability": "cap_on_liability",
    "Non-Compete": "non_compete",
    "IP Ownership Assignment": "ip_ownership_assignment",
    "Audit Rights": "audit_rights",
    "Anti-Assignment": "anti_assignment",
    "Change Of Control": "change_of_control",
    "Arbitration": "arbitration",
    "Indemnification": "indemnification",
    "Uncapped Liability": "uncapped_liability",
    "Liquidated Damages": "liquidated_damages",
    "Insurance": "insurance",
    "Non-Disparagement": "non_disparagement",
    "No-Solicitation Of Customers": "no_solicitation",
    "Exclusivity": "exclusivity",
    "Revenue/Profit Sharing": "revenue_profit_sharing",
    "Minimum Commitment": "minimum_commitment",
    "License Grant": "license_grant",
    "Warranty Duration": "warranty_duration",
    "Most Favored Nation": "most_favored_nation",
    "Price Restrictions": "price_restrictions",
    "Volume Restriction": "volume_restriction",
    "Joint IP Ownership": "joint_ip_ownership",
    "Source Code Escrow": "source_code_escrow",
    "Post-Termination Services": "post_termination_services",
    "Unlimited/All-You-Can-Eat-License": "unlimited_all_you_can_eat_license",
    "Irrevocable Or Perpetual License": "irrevocable_or_perpetual_license",
    "Affiliate License-Licensor": "affiliate_license_licensor",
    "Affiliate License-Licensee": "affiliate_license_licensee",
    "Class Action Waiver": "class_action_waiver",
    "Third Party Beneficiary": "third_party_beneficiary",
    "Covenant Not To Sue": "covenant_not_to_sue",
}


def load_cuad_ground_truth(
    contracts: list[dict],
    cuad_json_path: str = "./data/CUADv1.json"
) -> dict:
    """
    Load ground truth directly from CUADv1.json.
    Uses the ground_truth field already attached to each contract dict
    during loading — no second file read needed if contracts were loaded
    via load_cuad_json().

    Falls back to re-reading CUADv1.json if ground_truth is missing.

    Returns: {contract_id: {question_fragment: bool}}
    """
    # Fast path: ground truth already attached during loading
    if contracts and contracts[0].get("ground_truth"):
        print("[eval] Using ground truth embedded in contract objects")
        result = {}
        for c in contracts:
            cid = c["contract_id"]
            gt = c.get("ground_truth", {})
            # Convert {question: [answer_texts]} → {question_keyword: bool}
            result[cid] = {
                q: len(answers) > 0 and any(a.strip() for a in answers)
                for q, answers in gt.items()
            }
        return result

    # Fallback: re-read CUADv1.json
    import json
    from pathlib import Path
    if not Path(cuad_json_path).exists():
        print(f"[eval] CUADv1.json not found at {cuad_json_path}. Skipping eval.")
        return {}

    print(f"[eval] Reading ground truth from {cuad_json_path}")
    with open(cuad_json_path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    contract_id_set = {c["contract_id"] for c in contracts}
    result = {}

    for entry in cuad.get("data", []):
        from ingest.extract import sanitize_id
        cid = sanitize_id(entry.get("title", ""))
        if cid not in contract_id_set:
            continue
        paragraphs = entry.get("paragraphs", [])
        if not paragraphs:
            continue
        gt = {}
        for qa in paragraphs[0].get("qas", []):
            question = qa.get("question", "")
            answers = qa.get("answers", [])
            answer_texts = [a["text"] for a in answers if a.get("text", "").strip()]
            gt[question] = len(answer_texts) > 0
        result[cid] = gt

    print(f"[eval] Ground truth loaded for {len(result)} contracts")
    return result


def _find_gt_for_clause(gt_clauses: dict, cuad_name: str) -> bool:
    """
    Match a CUAD category name against question strings in ground truth.
    CUAD questions follow the pattern: "Does the clause contain [Category]?"
    We match by checking if the category name appears in any question key.
    """
    cuad_name_lower = cuad_name.lower()
    for question, is_present in gt_clauses.items():
        if cuad_name_lower in question.lower():
            return bool(is_present)
    return False


def compute_metrics(
    extracted: dict,
    ground_truth: dict
) -> dict:
    """
    Compute precision, recall, F1 per clause category.

    extracted: {contract_id: {clause_key: {present: bool, ...}}}
    ground_truth: {contract_id: {question_string: bool}}
      — question_string comes from CUADv1.json qas[].question

    Returns: {category: {precision, recall, f1, tp, fp, fn, support}}
    """
    category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for contract_id in extracted:
        if contract_id not in ground_truth:
            continue

        our_clauses = extracted[contract_id]
        gt_clauses = ground_truth[contract_id]

        for cuad_name, clause_key in CUAD_CATEGORY_MAP.items():
            our_present = our_clauses.get(clause_key, {}).get("present", False)
            gt_present = _find_gt_for_clause(gt_clauses, cuad_name)

            if our_present and gt_present:
                category_stats[clause_key]["tp"] += 1
            elif our_present and not gt_present:
                category_stats[clause_key]["fp"] += 1
            elif not our_present and gt_present:
                category_stats[clause_key]["fn"] += 1
            else:
                category_stats[clause_key]["tn"] += 1

    results = {}
    overall_tp = overall_fp = overall_fn = 0

    for clause_key, stats in category_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        results[clause_key] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn
        }

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall /
                  (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0)

    results["_overall"] = {
        "precision": round(overall_precision, 3),
        "recall": round(overall_recall, 3),
        "f1": round(overall_f1, 3),
        "tp": overall_tp, "fp": overall_fp, "fn": overall_fn
    }

    return results


def print_eval_report(metrics: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "="*65)
    print("CUAD EXTRACTION EVALUATION REPORT")
    print("="*65)
    print(f"{'Category':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>5}")
    print("-"*65)

    overall = metrics.pop("_overall", None)
    for cat, m in sorted(metrics.items(), key=lambda x: -x[1]["f1"]):
        print(f"{cat:<35} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>5}")

    if overall:
        print("-"*65)
        print(f"{'OVERALL':<35} {overall['precision']:>6.3f} "
              f"{overall['recall']:>6.3f} {overall['f1']:>6.3f}")
        metrics["_overall"] = overall
    print("="*65)


def run_evaluation(
    extracted: dict,
    contracts: list = None,
    cuad_json_path: str = "./data/CUADv1.json",
    save_path: str = "./outputs/eval_results.json"
) -> dict:
    """
    Full evaluation pipeline.
    extracted:      {contract_id: clause_dict}
    contracts:      original list of contract dicts (with embedded ground_truth)
    cuad_json_path: fallback path to CUADv1.json if ground_truth not embedded
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(extracted[list(extracted.keys())[0]])
    ground_truth = load_cuad_ground_truth(
        contracts or [],
        cuad_json_path=cuad_json_path
    )

    if not ground_truth:
        print("[eval] No ground truth available. Skipping evaluation.")
        return {}

    metrics = compute_metrics(extracted, ground_truth)
    print_eval_report(metrics)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Results saved to {save_path}")

    return metrics