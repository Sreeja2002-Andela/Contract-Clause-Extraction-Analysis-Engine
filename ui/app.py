"""
ui/app.py — Streamlit frontend for the Contract Analysis Engine.

Run with: streamlit run ui/app.py

Four views:
1. Dashboard    — Risk overview across all contracts
2. Contract     — Clause viewer for a single contract
3. Compare      — Side-by-side clause comparison
4. Ask          — Natural language QA chat
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import httpx
import pandas as pd
from extraction.clause_extractor import CLAUSE_CATEGORIES

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Contract Analysis Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def risk_badge(level: str) -> str:
    colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
    return colors.get(level, "⚪")


def clause_display_name(key: str) -> str:
    return key.replace("_", " ").title()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Contract Analyzer")
    st.caption("CUAD-powered clause extraction")

    st.divider()

    view = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📋 Contract View", "⚖️ Compare Clauses", "💬 Ask a Question"],
        label_visibility="collapsed"
    )

    st.divider()

    with st.expander("⚙️ Load Contracts"):
        source = st.selectbox(
            "Data source",
            ["auto", "json", "local", "huggingface"],
            help="auto = uses CUADv1.json if present, then local files, then HuggingFace"
        )
        max_c = st.slider("Max contracts", 5, 510, 20)
        cuad_json = st.text_input("CUADv1.json path", "./data/CUADv1.json")

        if st.button("🚀 Start Ingestion", use_container_width=True):
            payload = {
                "source": source,
                "max_contracts": max_c,
                "cuad_json": cuad_json
            }
            result = api_post("/ingest", payload)
            if result:
                st.success("Ingestion started! Refresh to see progress.")

    status = api_get("/ingest-status")
    if status:
        s = status["status"]
        emoji = {"idle": "⚪", "running": "🔄", "done": "✅", "error": "❌"}.get(s, "⚪")
        st.caption(f"{emoji} Status: **{s}** | {status['contracts_loaded']} contracts loaded")
        if s == "running" and st.button("🔄 Refresh"):
            st.rerun()


# ── Dashboard View ────────────────────────────────────────────────────────────

if view == "🏠 Dashboard":
    st.header("Risk Dashboard")

    risk_data = api_get("/risk-summary")
    if not risk_data or not risk_data.get("contracts"):
        st.info("No contracts loaded yet. Use the sidebar to load contracts.")
        st.stop()

    contracts = risk_data["contracts"]
    total = risk_data["total_contracts"]
    high_risk = risk_data["high_risk_count"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Contracts", total)
    col2.metric("🔴 High Risk", high_risk)
    col3.metric("🟡 Medium Risk", sum(1 for c in contracts if c["overall_risk"] == "MEDIUM"))
    col4.metric("🟢 Low Risk", sum(1 for c in contracts if c["overall_risk"] == "LOW"))

    st.divider()

    df_rows = []
    for c in contracts:
        df_rows.append({
            "Contract": c["contract_id"][:40] + ("..." if len(c["contract_id"]) > 40 else ""),
            "Risk": f"{risk_badge(c['overall_risk'])} {c['overall_risk']}",
            "Score": c["risk_score"],
            "🔴 High": c["high_count"],
            "🟡 Medium": c["medium_count"],
            "🟢 Low": c["low_count"],
            "Summary": c["summary"]
        })

    df = pd.DataFrame(df_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("High Risk Contracts — Top Issues")
    for c in contracts:
        if c["overall_risk"] == "HIGH":
            with st.expander(f"🔴 {c['contract_id'][:60]}"):
                for flag in c.get("top_flags", []):
                    st.error(f"**{clause_display_name(flag['clause'])}**: {flag['reason']}")
                    st.caption(f"💡 {flag['recommendation']}")


# ── Contract View ────────────────────────────────────────────────────────────

elif view == "📋 Contract View":
    st.header("Contract Clause Viewer")

    contracts_data = api_get("/contracts")
    if not contracts_data or not contracts_data.get("contracts"):
        st.info("No contracts loaded yet.")
        st.stop()

    contract_options = {
        f"{risk_badge(c['overall_risk'])} {c['contract_id'][:50]}": c["contract_id"]
        for c in contracts_data["contracts"]
    }

    selected_label = st.selectbox("Select contract", list(contract_options.keys()))
    selected_id = contract_options[selected_label]

    detail = api_get(f"/contract/{selected_id}")
    if not detail:
        st.stop()

    risk = detail["risk_analysis"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Risk", f"{risk_badge(risk['overall_risk'])} {risk['overall_risk']}")
    col2.metric("Clauses Found", detail["total_clauses_found"])
    col3.metric("Clauses Absent", detail["total_clauses_absent"])

    if risk.get("flags"):
        with st.expander(f"⚠️ {len(risk['flags'])} Risk Flags", expanded=True):
            for flag in risk["flags"]:
                level_fn = {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.info}
                fn = level_fn.get(flag["level"], st.info)
                fn(f"**{clause_display_name(flag['clause'])}** — {flag['reason']}\n\n💡 _{flag['recommendation']}_")

    st.divider()
    st.subheader("Extracted Clauses")

    present = detail.get("present_clauses", {})
    if not present:
        st.warning("No clauses extracted yet.")
    else:
        for clause_key, clause_data in present.items():
            with st.expander(f"✅ {clause_display_name(clause_key)}"):
                if clause_data.get("page"):
                    st.caption(f"📍 Page ~{clause_data['page']}")
                st.markdown(f"> {clause_data.get('text', 'No text extracted')}")

    st.subheader("Absent Clauses")
    absent = detail.get("absent_clauses", [])
    if absent:
        absent_cols = st.columns(3)
        for i, clause_key in enumerate(absent):
            absent_cols[i % 3].markdown(f"❌ {clause_display_name(clause_key)}")


# ── Compare View ─────────────────────────────────────────────────────────────

elif view == "⚖️ Compare Clauses":
    st.header("Cross-Contract Clause Comparison")

    contracts_data = api_get("/contracts")
    if not contracts_data or not contracts_data.get("contracts"):
        st.info("No contracts loaded yet.")
        st.stop()

    all_ids = [c["contract_id"] for c in contracts_data["contracts"]]
    selected_ids = st.multiselect(
        "Select contracts to compare",
        all_ids,
        default=all_ids[:min(5, len(all_ids))]
    )


    clause_options = [clause_display_name(c) for c in CLAUSE_CATEGORIES]
    clause_display = st.selectbox("Clause to compare", clause_options)
    clause_key = CLAUSE_CATEGORIES[clause_options.index(clause_display)]

    if selected_ids and st.button("Compare", use_container_width=True):
        result = api_get("/compare", {
            "clause": clause_key,
            "contract_ids": ",".join(selected_ids)
        })

        if result:
            col1, col2 = st.columns(2)
            col1.metric("Present in", f"{result['present_in']} / {result['contracts_compared']}")
            col2.metric("Absent in", result["absent_in"])

            st.divider()
            for item in result["comparison"]:
                if item["present"]:
                    with st.expander(f"✅ {item['contract_id'][:50]}", expanded=False):
                        if item.get("page"):
                            st.caption(f"📍 Page ~{item['page']}")
                        st.markdown(f"> {item.get('text', 'No text extracted')}")
                else:
                    st.markdown(f"❌ **{item['contract_id'][:50]}** — clause absent")


# ── QA View ──────────────────────────────────────────────────────────────────

elif view == "💬 Ask a Question":
    st.header("Ask About a Contract")

    contracts_data = api_get("/contracts")
    if not contracts_data or not contracts_data.get("contracts"):
        st.info("No contracts loaded yet.")
        st.stop()

    contract_options = {
        c["contract_id"][:60]: c["contract_id"]
        for c in contracts_data["contracts"]
    }

    selected_label = st.selectbox("Select contract", list(contract_options.keys()))
    selected_id = contract_options[selected_label]

    st.caption("Example questions: Does this contract have a non-compete clause? What is the governing law? Is there a cap on liability?")

    question = st.text_input("Your question", placeholder="Does this contract have a termination for convenience clause?")

    if question and st.button("Ask", use_container_width=True):
        with st.spinner("Retrieving relevant sections and generating answer..."):
            result = api_post("/qa", {"contract_id": selected_id, "question": question})

        if result:
            st.subheader("Answer")
            st.markdown(result["answer"])

            if result.get("sources"):
                with st.expander(f"📚 Sources ({len(result['sources'])} retrieved sections)"):
                    for i, src in enumerate(result["sources"]):
                        st.markdown(f"**Source {i+1}** — Page ~{src.get('page', '?')} | Score: {src.get('rrf_score', 0):.4f}")
                        st.markdown(f"> {src['text']}")
                        st.divider()