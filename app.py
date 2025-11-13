import os
import io
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import html

import streamlit as st
import threading
import pandas as pd

# ============= Page Config =============
st.set_page_config(
    page_title="Contract Insight — Clause & Risk Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============= Theming / Glossy UI =============
GLASS_CSS = """
<style>
:root { --glass-bg: rgba(255, 255, 255, 0.08); --glass-br: 18px; --glass-bd: 1px solid rgba(255,255,255,.15); }
/* full-page gradient */
.stApp {
  background: radial-gradient(1200px 800px at 10% 10%, #3b82f6 0%, rgba(59,130,246,0) 40%),
              radial-gradient(800px 600px at 90% 20%, #22d3ee 0%, rgba(34,211,238,0) 35%),
              radial-gradient(900px 700px at 40% 90%, #a78bfa 0%, rgba(167,139,250,0) 40%),
              linear-gradient(180deg, #0b1220 0%, #0e1426 100%);
  color: #e5e7eb;
}
/* glass cards */
.block-container { padding-top: 2.2rem; }
.glass {
  background: var(--glass-bg);
  border: var(--glass-bd);
  border-radius: var(--glass-br);
  box-shadow: 0 20px 40px rgba(0,0,0,.35);
  backdrop-filter: blur(10px);
}
.glass.pad { padding: 1.25rem 1.25rem; }
/* chips */
.chip { display:inline-flex; align-items:center; gap:.5rem; padding:.25rem .6rem; font-weight:600; border-radius:999px; border:1px solid rgba(255,255,255,.2); background:rgba(255,255,255,.06); }
.chip .dot{width:.55rem;height:.55rem;border-radius:50%;display:inline-block}
.chip.low .dot{background:#22c55e}
.chip.medium .dot{background:#f59e0b}
.chip.high .dot{background:#ef4444}
.chip.critical .dot{background:#ef4444}
/* clause box */
.clause { padding:1rem; border-radius:14px; border:1px solid rgba(255,255,255,.1); background:rgba(255,255,255,.03); }
.highlight { background: rgba(239, 68, 68, .2); padding: 0 .25rem; border-radius: 6px; }
/* metrics */
.metric { display:flex; flex-direction:column; padding:1rem; border-radius:18px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.15); }
.metric .label { opacity:.8; font-size:.85rem; }
.metric .value { font-size:1.6rem; font-weight:800; margin-top:.25rem; }
/* buttons */
button[kind="primary"] { background: linear-gradient(135deg, #22d3ee, #3b82f6); border:none; }
/* tables */
thead tr th { background: rgba(255,255,255,.06) !important; }
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ============= Helper Utilities =============
SEVERITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

def chip(level: str) -> str:
    lvl = (level or "").capitalize()
    cls = "low" if lvl == "Low" else "medium" if lvl == "Medium" else "high" if lvl == "High" else "critical" if lvl == "Critical" else ""
    return f'<span class="chip {cls}"><span class="dot"></span>{lvl or "N/A"}</span>'


def call_backend(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Analyze contract using fast analysis pipeline"""
    import requests  # local import to keep startup fast
    api_url = os.getenv("BACKEND_API_URL")  # Use separate backend URL
    
    # If Backend API URL is set, use backend API
    if api_url:
        files = {"file": (filename, file_bytes, "application/pdf")}
        try:
            with st.spinner("Analyzing your contract in the backend…"):
                resp = requests.post(api_url.rstrip("/") + "/analyze", files=files, timeout=120)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            st.error(f"Backend error: {e}. Using local fast analysis instead.")
    
    # Use local fast analysis (Phase 1 triage)
    try:
        from fast_analysis import analyze_contract_fast
        with st.spinner("Analyzing contract (fast mode)…"):
            result = analyze_contract_fast(file_bytes, filename)
            # Check if result has error
            if result.get("error"):
                st.warning(f"Analysis completed with warnings: {result.get('error')}. Showing partial results.")
            return result
    except ImportError as e:
        st.error(f"Missing required module: {e}. Please install dependencies: pip install -r requirements.txt")
        return mock_analysis()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Analysis error: {str(e)}. Showing demo data.")
        st.expander("Error Details").code(error_details)
        return mock_analysis()


def mock_analysis() -> Dict[str, Any]:
    return {
        "summary": {
            "contract_type": "Master Service Agreement",
            "party_names": ["Acme Corp", "Globex LLC"],
            "effective_date": "2024-04-15",
            "expiry_date": "2026-04-14",
        },
        "overall_risk_score": 71,
        "clauses": [
            {
                "id": "cl_001",
                "title": "Indemnification",
                "text": "Supplier shall indemnify and hold harmless Client from and against any and all claims, liabilities, damages, and expenses arising out of or relating to the Services. This indemnity is unlimited in amount.",
                "risk": "High",
                "tags": ["Liability", "Indemnity"],
                "problematic_spans": ["unlimited in amount"],
            },
            {
                "id": "cl_002",
                "title": "Limitation of Liability",
                "text": "In no event shall either party be liable for indirect or consequential damages. Total aggregate liability shall not exceed the fees paid in the preceding 6 months.",
                "risk": "Medium",
                "tags": ["Liability", "Damages"],
                "problematic_spans": ["6 months"],
            },
            {
                "id": "cl_003",
                "title": "Termination for Convenience",
                "text": "Client may terminate this Agreement for convenience upon 15 days' written notice without penalty.",
                "risk": "Low",
                "tags": ["Termination"],
                "problematic_spans": [],
            },
        ],
        "risks": [
            {
                "id": "r_101",
                "severity": "High",
                "title": "Unlimited indemnity exposure",
                "description": "Indemnity clause does not cap exposure, creating open-ended liability.",
                "clause_id": "cl_001",
                "suggestions": [
                    "Cap indemnity to a monetary limit (e.g., 12 months' fees)",
                    "Exclude third-party IP claims unless expressly negotiated",
                ],
            },
            {
                "id": "r_102",
                "severity": "Medium",
                "title": "Low liability cap period",
                "description": "Cap based on fees in preceding 6 months may be insufficient for longer engagements.",
                "clause_id": "cl_002",
                "suggestions": [
                    "Increase cap to 12–24 months' fees",
                    "Add carve-outs for gross negligence, fraud, data breach",
                ],
            },
        ],
    }


def highlight_spans(text: str, spans: List[str]) -> str:
    out = text
    for s in sorted(set(spans or []), key=len, reverse=True):
        out = out.replace(s, f'<span class="highlight">{s}</span>')
    return out


def to_download_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2).encode("utf-8")


# ============= Sidebar =============
st.sidebar.title("📄 Contract Insight")
st.sidebar.caption("AI-powered clause & risk analysis")
nav = st.sidebar.radio("Navigation", ["Upload PDF", "Dashboard", "Ask Questions"], index=0, help="Start by uploading a contract PDF, then view insights or ask questions.")

with st.sidebar:
    st.markdown("---")
    st.caption("Backend Configuration")
    
    # Backend API URL input
    st.text_input(
        "Backend API URL (optional)", 
        key="backend_api_url", 
        value=os.getenv("BACKEND_API_URL", ""), 
        help="URL for backend analysis service. Leave empty to use local analysis.",
        placeholder="http://localhost:8000"
    )
    if st.session_state.get("backend_api_url"):
        os.environ["BACKEND_API_URL"] = st.session_state["backend_api_url"]
    
    # Gemini API Key input
    st.text_input(
        "Gemini API Key (optional)",
        key="gemini_api_key",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="For enhanced QA responses. Leave empty to use fallback methods.",
        type="password",
        placeholder="AIza..."
    )
    if st.session_state.get("gemini_api_key"):
        os.environ["GEMINI_API_KEY"] = st.session_state["gemini_api_key"]
    
    st.markdown("---")
    st.caption("Export")
    if "analysis" in st.session_state:
        st.download_button(
            "Download JSON report",
            data=to_download_bytes(st.session_state["analysis"]),
            file_name=f"contract_insight_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )

# ============= Upload Page =============
if nav == "Upload PDF":
    st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
    st.subheader("Upload a contract or agreement")
    st.caption("PDF only • Max ~50 MB (backend dependent)")

    uploaded = st.file_uploader("Drag & drop or browse your PDF", type=["pdf"], label_visibility="collapsed")

    colA, colB = st.columns([1, 2])
    with colA:
        analyze_clicked = st.button("Analyze Contract", type="primary", use_container_width=True, disabled=uploaded is None)
    with colB:
        st.info("Your file is sent to the backend you configure above. If none is set, we'll show a realistic demo.")

    if analyze_clicked and uploaded is not None:
        # Read file bytes (reset file pointer first)
        uploaded.seek(0)
        file_bytes = uploaded.read()
        
        # Verify file has content
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty. Please upload a valid PDF file.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            result = call_backend(file_bytes, uploaded.name)
            st.session_state["analysis"] = result
            # Save uploaded bytes for background refinement (copy bytes)
            st.session_state["_uploaded_bytes"] = file_bytes
            st.session_state["_uploaded_name"] = uploaded.name
            # Kick off background refinement (Phase 2)
            def _run_refinement():
                try:
                    # Suppress Streamlit context warnings for background thread
                    import logging
                    logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
                    
                    from fast_analysis import refine_analysis
                    # Get fresh copy of bytes from session state
                    bytes_copy = st.session_state.get("_uploaded_bytes", b"")
                    name_copy = st.session_state.get("_uploaded_name", "uploaded.pdf")
                    
                    if len(bytes_copy) == 0:
                        logging.warning(f"No file bytes available for refinement (file: {name_copy})")
                        return
                    
                    logging.info(f"Starting refinement for {name_copy} ({len(bytes_copy)} bytes)")
                    refined = refine_analysis(bytes_copy, name_copy)
                    
                    if refined and not refined.get("error"):
                        # Use session state directly without st calls
                        st.session_state["analysis"] = refined
                        st.session_state["refinement_ready"] = True
                        logging.info(f"Refinement completed successfully for {name_copy}")
                    else:
                        logging.warning(f"Refinement returned error or empty result for {name_copy}")
                except Exception as e:
                    import logging
                    import traceback
                    logging.error(f"Refinement failed: {e}")
                    logging.error(traceback.format_exc())
                    st.session_state["refinement_ready"] = False
            # Reset refinement flag for new upload
            st.session_state["_refine_thread_started"] = False
            st.session_state["refinement_ready"] = False
            
            # Start refinement thread
            threading.Thread(target=_run_refinement, daemon=True).start()
            st.session_state["_refine_thread_started"] = True
            st.success("Analysis complete. Head to the **Dashboard**.")
            st.markdown("</div>", unsafe_allow_html=True)

# ============= Q&A Page =============
if nav == "Ask Questions":
    if "analysis" not in st.session_state:
        st.warning("Please upload and analyze a PDF first on the **Upload PDF** page.")
    else:
        st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
        st.subheader("💬 Ask Questions About Your Contract")
        st.caption("Ask natural language questions about the uploaded contract. Answers include citations and confidence scores.")
        
        # Initialize QA engine if not already done or if API key changed
        current_gemini_key = os.getenv("GEMINI_API_KEY")
        if ("qa_engine" not in st.session_state or 
            st.session_state.get("qa_engine_api_key") != current_gemini_key):
            try:
                import sys
                import os
                from pathlib import Path
                
                # Ensure src is in Python path for imports
                workspace_path = Path.cwd()
                src_path = workspace_path / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                # Try importing with better error handling
                try:
                    from src.provenance_qa import create_qa_engine
                    from src.provenance_qa.models.context_models import DocumentChunk, ChunkType, RetrievalSource, ContextStrategy
                except ImportError as import_err:
                    # Try alternative import path
                    import importlib.util
                    qa_init_path = src_path / "provenance_qa" / "__init__.py"
                    if qa_init_path.exists():
                        spec = importlib.util.spec_from_file_location("provenance_qa", qa_init_path)
                        provenance_qa_module = importlib.util.module_from_spec(spec)
                        sys.modules["provenance_qa"] = provenance_qa_module
                        spec.loader.exec_module(provenance_qa_module)
                        create_qa_engine = provenance_qa_module.create_qa_engine
                        from src.provenance_qa.models.context_models import DocumentChunk, ChunkType, RetrievalSource, ContextStrategy
                    else:
                        raise import_err
                
                current_gemini_key = os.getenv("GEMINI_API_KEY")
                if not current_gemini_key:
                    st.info("💡 Tip: Enter your Gemini API Key in the sidebar for enhanced answers. Working in fallback mode.")
                
                qa_engine = create_qa_engine(
                    gemini_api_key=current_gemini_key,
                    workspace_path=str(workspace_path),
                    enable_validation=True,
                    context_strategy=ContextStrategy.FOCUSED
                )
                
                # Enhance QA engine to use uploaded document's clauses with risk assessment context
                analysis_data = st.session_state.get("analysis", {})
                if analysis_data and "clauses" in analysis_data:
                    # Create enhanced document chunks with risk assessment details
                    document_chunks = []
                    
                    # Add document summary as first chunk for context
                    summary = analysis_data.get("summary", {})
                    overall_risk = analysis_data.get("overall_risk_score", 0)
                    
                    summary_content = f"""DOCUMENT SUMMARY:
Contract Type: {summary.get('contract_type', 'Unknown')}
Parties: {', '.join(summary.get('party_names', ['Unknown']))}
Effective Date: {summary.get('effective_date', 'Not specified')}
Expiry Date: {summary.get('expiry_date', 'Not specified')}
Overall Risk Score: {overall_risk}/100

This document contains {len(analysis_data.get('clauses', []))} clauses with risk assessments.
"""
                    
                    from src.provenance_qa.models.context_models import DocumentChunk, ChunkType, RetrievalSource
                    document_chunks.append(DocumentChunk(
                        chunk_id="doc_summary",
                        document_id="uploaded_document",
                        document_title=summary.get("contract_type", "Uploaded Contract"),
                        content=summary_content,
                        chunk_type=ChunkType.METADATA,
                        retrieval_source=RetrievalSource.HYBRID,
                        relevance_score=1.0,
                        similarity_score=1.0,
                        section_title="Document Summary",
                        page_number=0,
                        paragraph_number=0
                    ))
                    
                    # Create enhanced chunks from clauses with risk assessment context
                    for clause in analysis_data["clauses"]:
                        # Build enriched content with risk assessment details
                        clause_text = clause.get("text", "")
                        risk_level = clause.get("risk", "Low")
                        clause_title = clause.get("title", "Unknown Clause")
                        tags = clause.get("tags", [])
                        problematic_spans = clause.get("problematic_spans", [])
                        
                        # Find corresponding risk details if available
                        clause_id = clause.get("id", "")
                        risk_details = None
                        for risk in analysis_data.get("risks", []):
                            if risk.get("clause_id") == clause_id:
                                risk_details = risk
                                break
                        
                        # Build enriched content
                        enriched_content = f"""CLAUSE: {clause_title}
Risk Level: {risk_level}
Tags: {', '.join(tags) if tags else 'None'}

{clause_text}"""
                        
                        # Add risk assessment details if available
                        if risk_details:
                            enriched_content += f"""

RISK ASSESSMENT:
Severity: {risk_details.get('severity', risk_level)}
Description: {risk_details.get('description', 'No detailed risk assessment available')}
Suggestions: {'; '.join(risk_details.get('suggestions', []))}"""
                        
                        # Add problematic spans if any
                        if problematic_spans:
                            enriched_content += f"\n\nKey Risk Indicators: {', '.join(problematic_spans[:3])}"
                        
                        chunk = DocumentChunk(
                            chunk_id=clause_id,
                            document_id="uploaded_document",
                            document_title=summary.get("contract_type", "Uploaded Contract"),
                            content=enriched_content,
                            chunk_type=ChunkType.PARAGRAPH,
                            retrieval_source=RetrievalSource.HYBRID,
                            relevance_score=0.9 if risk_level in ["High", "Medium"] else 0.7,
                            similarity_score=0.8,
                            section_title=clause_title,
                            page_number=1,
                            paragraph_number=1,
                            metadata={
                                "risk_level": risk_level,
                                "tags": tags,
                                "clause_type": clause_title,
                                "has_risk_details": risk_details is not None
                            }
                        )
                        document_chunks.append(chunk)
                    
                    # Store chunks for QA engine to use
                    st.session_state["document_chunks"] = document_chunks
                    st.session_state["qa_engine"] = qa_engine
                    st.session_state["qa_engine_api_key"] = current_gemini_key
                else:
                    st.session_state["qa_engine"] = qa_engine
                    st.session_state["qa_engine_api_key"] = current_gemini_key
                    
            except Exception as e:
                st.error(f"Failed to initialize QA engine: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state["qa_engine"] = None
        
        # Refresh document chunks if refined results available
        try:
            if st.session_state.get("refinement_ready") and st.session_state.get("qa_engine"):
                analysis_data = st.session_state.get("analysis", {})
                if analysis_data and "clauses" in analysis_data:
                    from src.provenance_qa.models.context_models import DocumentChunk, ChunkType, RetrievalSource
                    
                    # Recreate enhanced chunks with updated data
                    document_chunks = []
                    
                    # Add document summary
                    summary = analysis_data.get("summary", {})
                    overall_risk = analysis_data.get("overall_risk_score", 0)
                    
                    summary_content = f"""DOCUMENT SUMMARY:
Contract Type: {summary.get('contract_type', 'Unknown')}
Parties: {', '.join(summary.get('party_names', ['Unknown']))}
Effective Date: {summary.get('effective_date', 'Not specified')}
Expiry Date: {summary.get('expiry_date', 'Not specified')}
Overall Risk Score: {overall_risk}/100

This document contains {len(analysis_data.get('clauses', []))} clauses with risk assessments.
"""
                    
                    document_chunks.append(DocumentChunk(
                        chunk_id="doc_summary",
                        document_id="uploaded_document",
                        document_title=summary.get("contract_type", "Uploaded Contract"),
                        content=summary_content,
                        chunk_type=ChunkType.METADATA,
                        retrieval_source=RetrievalSource.HYBRID,
                        relevance_score=1.0,
                        similarity_score=1.0,
                        section_title="Document Summary",
                        page_number=0,
                        paragraph_number=0
                    ))
                    
                    # Add enhanced clause chunks
                    for i, clause in enumerate(analysis_data["clauses"]):
                        clause_text = clause.get("text", "")
                        risk_level = clause.get("risk", "Low")
                        clause_title = clause.get("title", "Unknown Clause")
                        tags = clause.get("tags", [])
                        problematic_spans = clause.get("problematic_spans", [])
                        
                        clause_id = clause.get("id", f"cl_{i}")
                        risk_details = None
                        for risk in analysis_data.get("risks", []):
                            if risk.get("clause_id") == clause_id:
                                risk_details = risk
                                break
                        
                        enriched_content = f"""CLAUSE: {clause_title}
Risk Level: {risk_level}
Tags: {', '.join(tags) if tags else 'None'}

{clause_text}"""
                        
                        if risk_details:
                            enriched_content += f"""

RISK ASSESSMENT:
Severity: {risk_details.get('severity', risk_level)}
Description: {risk_details.get('description', 'No detailed risk assessment available')}
Suggestions: {'; '.join(risk_details.get('suggestions', []))}"""
                        
                        if problematic_spans:
                            enriched_content += f"\n\nKey Risk Indicators: {', '.join(problematic_spans[:3])}"
                        
                        document_chunks.append(DocumentChunk(
                            chunk_id=clause_id,
                            document_id="uploaded_document",
                            document_title=summary.get("contract_type", "Uploaded Contract"),
                            content=enriched_content,
                            chunk_type=ChunkType.PARAGRAPH,
                            retrieval_source=RetrievalSource.HYBRID,
                            relevance_score=0.9 if risk_level in ["High", "Medium"] else 0.7,
                            similarity_score=0.85,
                            section_title=clause_title,
                            page_number=1,
                            paragraph_number=i+1,
                            metadata={
                                "risk_level": risk_level,
                                "tags": tags,
                                "clause_type": clause_title,
                                "has_risk_details": risk_details is not None
                            }
                        ))
                    
                    st.session_state["document_chunks"] = document_chunks
        except Exception:
            pass

        # Question input
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the termination clauses? What are the payment terms?",
            key="qa_question"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask", type="primary", use_container_width=True)
        
        # Process question
        if ask_button and question:
            if not st.session_state.get("qa_engine"):
                st.error("QA engine not available. Please check system configuration.")
            else:
                with st.spinner("Processing your question…"):
                    try:
                        # Get document chunks from uploaded document
                        document_chunks = st.session_state.get("document_chunks", [])
                        response = st.session_state["qa_engine"].ask_question(
                            question,
                            document_chunks=document_chunks if document_chunks else None
                        )
                        
                        # Safely get answer text
                        answer_text = getattr(response, 'answer', '')
                        if not answer_text:
                            answer_text = getattr(response, 'summary', 'No answer generated.')
                        
                        # Display answer
                        st.markdown("### 📝 Answer")
                        st.markdown(answer_text if answer_text else "No answer available.")
                        
                        # Display metadata
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            confidence = getattr(response, 'overall_confidence', 0.0)
                            st.metric("Confidence", f"{confidence:.2f}")
                        with col_b:
                            quality = getattr(response, 'answer_quality', 0.0)
                            st.metric("Quality", f"{quality:.2f}")
                        with col_c:
                            citations_count = len(getattr(response, 'citations', []))
                            st.metric("Citations", citations_count)
                        
                        # Display citations
                        citations = getattr(response, 'citations', [])
                        if citations:
                            st.markdown("### 📚 Sources")
                            for i, citation in enumerate(citations[:5], 1):
                                source_title = getattr(citation, 'source_title', getattr(citation, 'document_title', f"Source {i}"))
                                cited_text = getattr(citation, 'cited_text', getattr(citation, 'relevant_text', ''))
                                relevance = getattr(citation, 'relevance_score', 0.0)
                                
                                # Escape HTML in text
                                source_title_escaped = html.escape(str(source_title))
                                cited_text_escaped = html.escape(str(cited_text[:200]))
                                
                                st.markdown(f"""
                                <div class='clause' style='margin-bottom:.8rem'>
                                    <div><strong>{i}. {source_title_escaped}</strong> <small>(Relevance: {relevance:.2f})</small></div>
                                    <div style='opacity:.9;margin-top:.35rem;font-size:.9em'>{cited_text_escaped}{'...' if len(cited_text) > 200 else ''}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No citations available for this answer.")
                        
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        
# ============= Dashboard Page =============
if nav == "Dashboard":
    if "analysis" not in st.session_state:
        st.warning("Please upload and analyze a PDF first on the **Upload PDF** page.")
    else:
        data = st.session_state["analysis"]

        # Show refinement status
        if st.session_state.get("refinement_ready"):
            st.success("Refined results loaded: clauses and risk scores updated.")
        else:
            st.info("Refining results in background… This will update automatically when ready.")

        # Header Card
        st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
        left, mid, right = st.columns([2.2, 1, 1])
        with left:
            st.markdown("### 🧭 Contract Overview")
            s = data.get("summary", {})
            parties = ", ".join(s.get("party_names", [])) or "—"
            meta_cols = st.columns(3)
            meta_cols[0].markdown(f"**Type**<br/>{s.get('contract_type','—')}", unsafe_allow_html=True)
            meta_cols[1].markdown(f"**Parties**<br/>{parties}", unsafe_allow_html=True)
            eff = s.get("effective_date", "—")
            exp = s.get("expiry_date", "—")
            meta_cols[2].markdown(f"**Term**<br/>{eff} → {exp}", unsafe_allow_html=True)
        with mid:
            st.markdown("<div class='metric'><span class='label'>Overall Risk Score</span><span class='value'>" + str(data.get("overall_risk_score", "—")) + "/100</span></div>", unsafe_allow_html=True)
        with right:
            total = len(data.get("clauses", []))
            high = sum(1 for c in data.get("clauses", []) if (c.get("risk") or "").lower() in ["high", "critical"])
            med = sum(1 for c in data.get("clauses", []) if (c.get("risk") or "").lower() == "medium")
            st.markdown(
                "<div class='metric'><span class='label'>Clauses (H/M)</span><span class='value'>" + f"{total} ({high}/{med})" + "</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("\n")

        # Two Columns: Risks & Clauses
        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
            st.subheader("⚠️ Key Risks")
            risks: List[Dict[str, Any]] = sorted(data.get("risks", []), key=lambda r: -SEVERITY_ORDER.get((r.get("severity") or "").capitalize(), 0))
            if not risks:
                st.caption("No risks detected.")
            for r in risks:
                st.markdown(
                    f"""
                    <div class='clause' style='margin-bottom:.8rem'>
                        <div style='display:flex;justify-content:space-between;align-items:center;gap:1rem'>
                            <div><strong>{r.get('title','Risk')}</strong></div>
                            <div>{chip(r.get('severity',''))}</div>
                        </div>
                        <div style='opacity:.9;margin-top:.35rem'>{r.get('description','')}</div>
                        <div style='margin-top:.5rem'>
                            <small style='opacity:.7'>Linked clause: <code>{r.get('clause_id','—')}</code></small>
                        </div>
                        {('<div style=\'margin-top:.6rem\'><strong>Suggested Mitigations</strong><ul>' + ''.join([f'<li>{html.escape(sug)}</li>' for sug in r.get('suggestions',[])]) + '</ul></div>') if r.get('suggestions') else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
            st.subheader("📑 Clauses")
            clauses = data.get("clauses", [])
            # Filters
            f1, f2 = st.columns(2)
            with f1:
                level = st.selectbox("Filter by Risk", ["All", "Low", "Medium", "High", "Critical"], index=0)
            with f2:
                query = st.text_input("Search text / title")

            filtered = []
            for c in clauses:
                if level != "All" and (c.get("risk") or "").capitalize() != level:
                    continue
                if query and (query.lower() not in (c.get("text","") + c.get("title","" )).lower()):
                    continue
                filtered.append(c)

            if not filtered:
                st.info("No clauses match your filters.")
            else:
                for c in filtered:
                    title = c.get("title", f"Clause {c.get('id')}")
                    tags = c.get("tags", [])
                    spans = c.get("problematic_spans", [])
                    text_html = highlight_spans(c.get("text", ""), spans)

                    tag_str = " ".join([f"<span class='chip'>{html.escape(t)}</span>" for t in tags])
                    st.markdown(
                        f"""
                        <div class='clause' style='margin-bottom:.9rem'>
                          <div style='display:flex;justify-content:space-between;gap:.75rem;align-items:center'>
                            <div><strong>{html.escape(title)}</strong> <small style='opacity:.6'>({html.escape(c.get('id','—'))})</small></div>
                            <div>{chip(c.get('risk',''))}</div>
                          </div>
                          <div style='margin-top:.4rem'>{text_html}</div>
                          <div style='margin-top:.6rem;display:flex;gap:.4rem;flex-wrap:wrap'>{tag_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("\n")

        # Tabular Views
        st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
        st.subheader("📊 Tabular View")
        tabs = st.tabs(["Risks Table", "Clauses Table"])
        with tabs[0]:
            r_df = pd.DataFrame(data.get("risks", []))
            if not r_df.empty:
                st.dataframe(r_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No risks to display.")
        with tabs[1]:
            c_df = pd.DataFrame(data.get("clauses", []))
            if not c_df.empty:
                st.dataframe(c_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No clauses to display.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Export Section
        st.markdown("<div class='glass pad'>", unsafe_allow_html=True)
        st.subheader("📥 Export & Sharing")
        colx, coly = st.columns([1,1])
        with colx:
            st.download_button(
                "Download JSON report",
                data=to_download_bytes(data),
                file_name=f"contract_insight_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with coly:
            # lightweight HTML export of the highlighted risks/clauses
            html_sections = [
                "<h2>Contract Insight — Summary</h2>",
                f"<p><b>Type:</b> {html.escape(str(data.get('summary',{}).get('contract_type','—')))}</p>",
                f"<p><b>Parties:</b> {html.escape(', '.join(data.get('summary',{}).get('party_names',[])) or '—')}</p>",
                f"<p><b>Term:</b> {html.escape(str(data.get('summary',{}).get('effective_date','—')))} → {html.escape(str(data.get('summary',{}).get('expiry_date','—')))}</p>",
                f"<p><b>Overall Risk Score:</b> {data.get('overall_risk_score','—')}/100</p>",
                "<hr/>",
                "<h3>Risks</h3>",
            ]
            for r in data.get("risks", []):
                html_sections.append(f"<p><b>{html.escape(str(r.get('title','Risk')))}</b> — {html.escape(str(r.get('severity','')))}<br/>{html.escape(str(r.get('description','')))}</p>")
            html_sections.append("<h3>Clauses</h3>")
            for c in data.get("clauses", []):
                html_sections.append(f"<h4>{html.escape(str(c.get('title','Clause')))}</h4><p>{html.escape(str(c.get('text','')))}</p>")
            html_doc = """
            <html><head><meta charset='utf-8'><title>Contract Insight Report</title></head>
            <body style='font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif; padding:24px; line-height:1.55;'>
            {body}
            </body></html>
            """.replace("{body}", "\n".join(html_sections))
            b = html_doc.encode("utf-8")
            st.download_button(
                "Download HTML report",
                data=b,
                file_name=f"contract_insight_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
