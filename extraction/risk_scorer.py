"""
risk_scorer.py — Risk flagging based on clause presence/absence.

Core insight: Absence of a clause IS the finding, not a retrieval failure.
A contract with no cap on liability is more dangerous than one with an
explicit uncapped liability clause — because the risk is hidden.

Risk levels: HIGH, MEDIUM, LOW, OK
"""

from dataclasses import dataclass


@dataclass
class RiskFlag:
    clause: str
    level: str        # HIGH, MEDIUM, LOW
    reason: str
    recommendation: str


RISK_RULES = [
    {
        "clause": "cap_on_liability",
        "condition": "absent",
        "level": "HIGH",
        "reason": "No cap on liability found. Party may be exposed to unlimited damages.",
        "recommendation": "Negotiate a liability cap, typically 12 months of fees paid."
    },
    {
        "clause": "uncapped_liability",
        "condition": "present",
        "level": "HIGH",
        "reason": "Explicit uncapped liability clause found.",
        "recommendation": "Negotiate mutual cap on liability before signing."
    },
    {
        "clause": "termination_for_convenience",
        "condition": "absent",
        "level": "HIGH",
        "reason": "No termination for convenience clause. Party may be locked into contract.",
        "recommendation": "Add mutual termination for convenience with 30-day notice."
    },
    {
        "clause": "governing_law",
        "condition": "absent",
        "level": "MEDIUM",
        "reason": "No governing law specified. Jurisdiction of disputes is unclear.",
        "recommendation": "Specify governing law and venue for dispute resolution."
    },
    {
        "clause": "indemnification",
        "condition": "present",
        "level": "MEDIUM",
        "reason": "Indemnification clause present — review scope carefully.",
        "recommendation": "Ensure indemnification is mutual and capped."
    },
    {
        "clause": "non_compete",
        "condition": "present",
        "level": "MEDIUM",
        "reason": "Non-compete clause found. May restrict future business activities.",
        "recommendation": "Review scope, duration, and geographic restrictions."
    },
    {
        "clause": "anti_assignment",
        "condition": "absent",
        "level": "LOW",
        "reason": "No anti-assignment clause. Contract rights may be freely transferred.",
        "recommendation": "Consider adding assignment restriction if partnership is key."
    },
    {
        "clause": "audit_rights",
        "condition": "absent",
        "level": "LOW",
        "reason": "No audit rights clause. Cannot verify compliance.",
        "recommendation": "Add audit rights for revenue-sharing or royalty arrangements."
    },
    {
        "clause": "arbitration",
        "condition": "present",
        "level": "LOW",
        "reason": "Arbitration clause found. Court litigation may be waived.",
        "recommendation": "Review arbitration rules, venue, and cost allocation."
    },
    {
        "clause": "class_action_waiver",
        "condition": "present",
        "level": "MEDIUM",
        "reason": "Class action waiver found. May limit remedies.",
        "recommendation": "Review enforceability in applicable jurisdiction."
    },
]


def score_contract(clauses: dict) -> dict:
    """
    Given extracted clauses for a contract, compute risk flags and overall score.

    Returns:
        {
          "overall_risk": "HIGH" | "MEDIUM" | "LOW",
          "risk_score": int (0-100, higher = riskier),
          "flags": [RiskFlag dicts],
          "summary": str
        }
    """
    flags = []

    for rule in RISK_RULES:
        clause_name = rule["clause"]
        clause_data = clauses.get(clause_name, {"present": False})
        is_present = clause_data.get("present", False)

        triggered = (
            (rule["condition"] == "present" and is_present) or
            (rule["condition"] == "absent" and not is_present)
        )

        if triggered:
            flags.append({
                "clause": clause_name,
                "level": rule["level"],
                "reason": rule["reason"],
                "recommendation": rule["recommendation"],
                "extracted_text": clause_data.get("text"),
                "page": clause_data.get("page")
            })

    high_count = sum(1 for f in flags if f["level"] == "HIGH")
    medium_count = sum(1 for f in flags if f["level"] == "MEDIUM")
    low_count = sum(1 for f in flags if f["level"] == "LOW")

    risk_score = min(100, high_count * 30 + medium_count * 15 + low_count * 5)

    if high_count >= 1:
        overall_risk = "HIGH"
    elif medium_count >= 2:
        overall_risk = "MEDIUM"
    else:
        overall_risk = "LOW"

    summary_parts = []
    if high_count:
        summary_parts.append(f"{high_count} high-risk issue{'s' if high_count > 1 else ''}")
    if medium_count:
        summary_parts.append(f"{medium_count} medium-risk issue{'s' if medium_count > 1 else ''}")
    if low_count:
        summary_parts.append(f"{low_count} low-risk note{'s' if low_count > 1 else ''}")

    summary = ", ".join(summary_parts) if summary_parts else "No significant risks detected"

    return {
        "overall_risk": overall_risk,
        "risk_score": risk_score,
        "flags": flags,
        "summary": summary,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count
    }


def score_all_contracts(contracts_clauses: dict) -> dict:
    """
    Score all contracts. Returns {contract_id: risk_result}
    contracts_clauses: {contract_id: clause_extraction_result}
    """
    return {
        cid: score_contract(clauses)
        for cid, clauses in contracts_clauses.items()
    }