"""
Fast Analysis Pipeline for Phase 1 Triage

This module implements fast document analysis using existing ContractSense modules
without heavy ML processing. Designed to complete in 3-8 seconds.

Components:
- Module 1: Fast document parsing (no OCR, minimal normalization)
- Clause detection: Heuristic clause boundary detection
- Risk assessment: Rule-based analyzers only (no ML)
- Output: Formatted results compatible with app.py
"""

import io
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Import ContractSense modules
from src.ingestion.document_parser import DocumentParser, ProcessingMode
from src.risk.analyzers.financial_analyzer import FinancialRiskAnalyzer
from src.risk.analyzers.legal_analyzer import LegalRiskAnalyzer

logger = logging.getLogger(__name__)


def analyze_contract_fast(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Fast analysis pipeline for uploaded contract
    
    Args:
        file_bytes: PDF file bytes
        filename: Original filename
        
    Returns:
        Dictionary with clauses, risks, and summary compatible with app.py format
    """
    start_time = time.time()
    
    try:
        # Step 1: Parse document in FAST mode (no OCR, minimal normalization)
        logger.info(f"Step 1: Fast document parsing for {filename}")
        
        # Verify input bytes
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Input file_bytes is empty for {filename}")
        logger.info(f"Received {len(file_bytes)} bytes for {filename}")
        
        # Create temp file and ensure it's written and closed before parsing
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        try:
            bytes_written = tmp_file.write(file_bytes)
            tmp_file.flush()  # Ensure data is written to disk
            tmp_path = Path(tmp_file.name)
            logger.info(f"Wrote {bytes_written} bytes to temp file: {tmp_path}")
        finally:
            tmp_file.close()  # Close file before parsing
        
        # Verify file exists and has content
        if not tmp_path.exists():
            raise FileNotFoundError(f"Temp file was not created: {tmp_path}")
        
        file_size = tmp_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Temp file is empty after write: {tmp_path}")
        if file_size != len(file_bytes):
            logger.warning(f"File size mismatch: wrote {len(file_bytes)} bytes but file is {file_size} bytes")
        
        try:
            parser = DocumentParser(
                enable_ocr=False,  # Skip OCR for speed
                enable_normalization=False,  # Skip normalization for speed
                enable_clause_detection=True,  # We need clause detection
                processing_mode=ProcessingMode.FAST
            )
            
            processed_doc = parser.parse(tmp_path)
            
            # Step 2: Extract clauses (already done by clause_detector in parser)
            logger.info(f"Step 2: Found {len(processed_doc.clauses)} clauses")
            
            # Step 3: Rule-based risk assessment (no ML)
            logger.info("Step 3: Rule-based risk assessment")
            risk_analyzer = RiskAnalyzerFast()
            
            clauses_data = []
            risks_data = []
            
            for clause in processed_doc.clauses:
                # Assess risk using rule-based analyzers only
                risk_assessment = risk_analyzer.assess_clause_fast(
                    clause_id=clause.clause_id,
                    clause_type=clause.clause_type or "unknown",
                    clause_text=clause.text
                )
                
                # Format clause data for app.py
                clause_data = {
                    "id": clause.clause_id,
                    "title": _format_clause_title(clause.clause_type or "Unknown Clause"),
                    "text": clause.text[:500] if len(clause.text) > 500 else clause.text,  # Limit length
                    "risk": risk_assessment["risk_level"],
                    "tags": _extract_tags(clause.clause_type, clause.text),
                    "problematic_spans": risk_assessment["problematic_spans"]
                }
                clauses_data.append(clause_data)
                
                # Create risk entry if high/medium risk
                if risk_assessment["risk_level"] in ["High", "Medium"]:
                    risk_data = {
                        "id": f"r_{clause.clause_id}",
                        "severity": risk_assessment["risk_level"],
                        "title": risk_assessment["risk_title"],
                        "description": risk_assessment["risk_description"],
                        "clause_id": clause.clause_id,
                        "suggestions": risk_assessment.get("suggestions", [])
                    }
                    risks_data.append(risk_data)
            
            # Step 4: Generate summary
            summary = _generate_summary(processed_doc, clauses_data)
            
            # Calculate overall risk score (0-100)
            overall_risk_score = _calculate_overall_risk_score(clauses_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Fast analysis completed in {processing_time:.2f}s")
            
            return {
                "summary": summary,
                "overall_risk_score": overall_risk_score,
                "clauses": clauses_data,
                "risks": risks_data,
                "metadata": {
                    "processing_time": processing_time,
                    "total_clauses": len(clauses_data),
                    "analysis_mode": "fast_triage"
                }
            }
            
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in fast analysis: {e}", exc_info=True)
        # Return error response
        return {
            "summary": {
                "contract_type": "Error",
                "party_names": [],
                "effective_date": None,
                "expiry_date": None,
            },
            "overall_risk_score": 0,
            "clauses": [],
            "risks": [],
            "error": str(e)
        }


def refine_analysis(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Phase 2 refinement: better clauses, ML risk if available, embeddings.
    Returns same schema as analyze_contract_fast(), enriched where possible.
    """
    start_time = time.time()
    try:
        # Verify input bytes
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Input file_bytes is empty for {filename}")
        logger.info(f"Refinement: Received {len(file_bytes)} bytes for {filename}")
        
        # Persist temp file for parser reuse
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        try:
            bytes_written = tmp_file.write(file_bytes)
            tmp_file.flush()  # Ensure data is written to disk
            tmp_path = Path(tmp_file.name)
            logger.info(f"Refinement: Wrote {bytes_written} bytes to temp file: {tmp_path}")
        finally:
            tmp_file.close()  # Close file before parsing
        
        # Verify file exists and has content
        if not tmp_path.exists():
            raise FileNotFoundError(f"Temp file was not created: {tmp_path}")
        
        file_size = tmp_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Temp file is empty after write: {tmp_path}")
        if file_size != len(file_bytes):
            logger.warning(f"Refinement: File size mismatch: wrote {len(file_bytes)} bytes but file is {file_size} bytes")

        try:
            # Parse with normalization enabled (still skip OCR for latency)
            parser = DocumentParser(
                enable_ocr=False,
                enable_normalization=True,
                enable_clause_detection=True,
                processing_mode=ProcessingMode.FAST
            )
            processed_doc = parser.parse(tmp_path)

            # Attempt ML-based clause extraction (graceful fallback)
            refined_clauses: List[Dict[str, Any]] = []
            used_ml = False
            try:
                from src.ml.inference.predictor import ClausePredictor
                from src.ml.models.bert_crf import BertCrfModel
                from src.ml.models.model_config import ModelConfig
                from src.ml.utils.tokenization import ContractTokenizer
                from src.ml.utils.label_encoding import LabelEncoder
                try:
                    import torch
                except ImportError:
                    logger.warning("PyTorch not available - skipping ML clause extraction")
                    raise ImportError("PyTorch required for ML extraction")
                # Load lightweight/default config; if resources missing, this will except
                model_cfg = ModelConfig()
                model = BertCrfModel(model_cfg)
                tokenizer = ContractTokenizer(model_cfg)
                label_encoder = LabelEncoder(model_cfg.label_list)
                predictor = ClausePredictor(model, tokenizer, label_encoder)

                doc_pred = predictor.predict_document(processed_doc.full_text, document_id=processed_doc.document_path)
                if doc_pred and doc_pred.predictions:
                    used_ml = True
                    for i, p in enumerate(doc_pred.predictions):
                        refined_clauses.append({
                            "id": f"ml_cl_{i:03d}",
                            "title": _format_clause_title(p.clause_type or "Clause"),
                            "text": p.text,
                            "risk": "Low",  # placeholder before risk scoring
                            "tags": _extract_tags(p.clause_type, p.text),
                            "problematic_spans": []
                        })
            except Exception:
                # Fall back to heuristic-detected clauses
                for clause in processed_doc.clauses:
                    refined_clauses.append({
                        "id": clause.clause_id,
                        "title": _format_clause_title(clause.clause_type or "Clause"),
                        "text": clause.text,
                        "risk": "Low",
                        "tags": _extract_tags(clause.clause_type, clause.text),
                        "problematic_spans": []
                    })

            # ML + rule-based risk assessment (graceful fallback to rules)
            try:
                from src.risk.risk_engine import RiskAssessmentEngine
                risk_engine = RiskAssessmentEngine()
                # risk_engine will use ML if model available; else rule-only
                clauses_for_risk = [{
                    "id": c["id"],
                    "type": (c.get("title") or "").lower().replace(" ", "_"),
                    "text": c["text"],
                } for c in refined_clauses]
                doc_risk = risk_engine.assess_document_risk(
                    document_id=processed_doc.document_path,
                    clauses=clauses_for_risk,
                    document_context=processed_doc.full_text
                )
                # Map risk back to clauses
                risk_map = {a.clause_id: a for a in doc_risk.clause_assessments}
                for c in refined_clauses:
                    a = risk_map.get(c["id"]) or next((risk_map.get(f"ml_{c['id']}") for _ in [0]), None)
                    if a:
                        level = a.final_risk_level.title()
                        c["risk"] = "High" if level.upper()=="HIGH" else ("Medium" if level.upper()=="MEDIUM" else "Low")
                # Build risks list for UI
                risks_ui: List[Dict[str, Any]] = []
                for a in doc_risk.clause_assessments:
                    if a.final_risk_level in ("HIGH", "MEDIUM"):
                        risks_ui.append({
                            "id": f"r_{a.clause_id}",
                            "severity": a.final_risk_level.title(),
                            "title": f"{a.final_risk_level.title()} risk — {a.clause_type.title()}",
                            "description": a.ml_rationale if a.ml_rationale else "Combined financial/legal risk factors detected.",
                            "clause_id": a.clause_id,
                            "suggestions": ["Review with counsel", "Negotiate favorable terms"]
                        })
            except Exception:
                # Fallback: keep existing low risks, no additional risk items
                risks_ui = []

            # Generate embeddings for refined clauses (Sentence-BERT fallback-safe)
            try:
                from src.vector_search.embedding_generator import EmbeddingConfig, SentenceTransformerModel, TextChunk, GeneratedEmbedding
                import numpy as np

                cfg = EmbeddingConfig(model_name="all-MiniLM-L6-v2", model_type="sentence_transformers", batch_size=32)
                model = SentenceTransformerModel(cfg)
                model.load_model()
                texts = [c["text"] for c in refined_clauses]
                embs = model.encode(texts)
                now = __import__("datetime").datetime.now()
                generated: List[GeneratedEmbedding] = []
                for i, c in enumerate(refined_clauses):
                    chunk = TextChunk(
                        content=c["text"],
                        chunk_id=c["id"],
                        chunk_type="clause",
                        chunk_index=i,
                        document_id=processed_doc.document_path,
                        source_file=filename,
                        metadata={"title": c.get("title", "")}
                    )
                    generated.append(GeneratedEmbedding(chunk=chunk, embedding=np.array(embs[i]), model_name=cfg.model_name, generation_time=now))
                # Return embeddings in metadata for session consumers (UI/QA)
                embeddings_payload = [{
                    "id": ge.chunk.chunk_id,
                    "text": ge.chunk.content,
                    "title": ge.chunk.metadata.get("title", ""),
                    "embedding": ge.embedding.tolist()
                } for ge in generated]
            except Exception:
                embeddings_payload = []

            # Summary and overall score
            summary = _generate_summary(processed_doc, refined_clauses)
            overall_risk_score = _calculate_overall_risk_score(refined_clauses)

            return {
                "summary": summary,
                "overall_risk_score": overall_risk_score,
                "clauses": refined_clauses,
                "risks": risks_ui,
                "metadata": {
                    "processing_time": time.time() - start_time,
                    "refined": True,
                    "used_ml_extraction": used_ml,
                    "embeddings": embeddings_payload
                }
            }
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Refinement failed: {e}", exc_info=True)
        return {
            "error": str(e)
        }


class RiskAnalyzerFast:
    """Fast rule-based risk analyzer (no ML)"""
    
    def __init__(self):
        self.financial_analyzer = FinancialRiskAnalyzer()
        self.legal_analyzer = LegalRiskAnalyzer()
        
        # Clause type multipliers for risk adjustment
        self.clause_multipliers = {
            "liability": 1.4,
            "indemnity": 1.3,
            "arbitration": 1.2,
            "payment": 1.0,
            "assignment": 0.9,
            "scope": 0.7,
            "limitation_of_liability": 1.4,
            "indemnification": 1.3,
            "dispute_resolution": 1.2,
            "termination": 1.0,
            "confidentiality": 0.9,
            "force_majeure": 0.8,
            "data_protection": 1.2,
            "governing_law": 1.1
        }
        
        # Major asymmetric keywords for high-risk trigger
        self.major_risk_keywords = [
            "sole discretion", "exclusive", "binding arbitration", "non-refundable", 
            "unlimited liability", "no reciprocal", "without cause", "immediate termination", 
            "no liability", "as is", "irrevocable", "perpetual", "unlimited", 
            "unrestricted", "absolute", "unconditional", "waive", "forfeit", "penalty"
        ]
        
        # Semantic calibration - phrases that force High risk
        self.critical_risk_phrases = [
            "unlimited liability", "broad indemnification", "absolute liability",
            "unconditional indemnification", "sole and exclusive remedy"
        ]
    
    def _detect_clause_theme(self, clause_text: str) -> str:
        """
        Pre-processing step to detect clause themes and assign correct legal category
        """
        text_lower = clause_text.lower()
        
        # Liability and indemnity patterns
        if any(kw in text_lower for kw in ["unlimited liability", "limitation of liability", "liable for", "damages", "loss"]):
            return "liability"
        if any(kw in text_lower for kw in ["indemnif", "hold harmless", "defend", "indemnity"]):
            return "indemnity"
        
        # Dispute resolution patterns
        if any(kw in text_lower for kw in ["arbitration", "arbitrator", "binding arbitration"]):
            return "arbitration"
        if any(kw in text_lower for kw in ["dispute resolution", "mediation", "litigation", "court"]):
            return "dispute_resolution"
        
        # Termination patterns
        if any(kw in text_lower for kw in ["terminat", "expire", "end", "breach", "default"]):
            return "termination"
        
        # Payment and financial patterns
        if any(kw in text_lower for kw in ["payment", "fee", "compensation", "invoice", "billing"]):
            return "payment"
        
        # Confidentiality and data protection
        if any(kw in text_lower for kw in ["confidential", "proprietary", "non-disclosure", "data protection", "privacy"]):
            if "data protection" in text_lower or "privacy" in text_lower:
                return "data_protection"
            return "confidentiality"
        
        # Force majeure patterns
        if any(kw in text_lower for kw in ["force majeure", "act of god", "unforeseeable", "beyond control"]):
            return "force_majeure"
        
        # Governing law patterns  
        if any(kw in text_lower for kw in ["governing law", "jurisdiction", "applicable law", "choice of law"]):
            return "governing_law"
        
        # Assignment patterns
        if any(kw in text_lower for kw in ["assignment", "assign", "transfer", "delegate"]):
            return "assignment"
        
        # Scope and definitions
        if any(kw in text_lower for kw in ["scope", "definition", "services", "deliverables", "work"]):
            return "scope"
        
        # Default to original clause_type or "general"
        return "general"
    
    def _count_major_risk_keywords(self, clause_text: str) -> int:
        """
        Count occurrences of major risk keywords in clause text
        """
        text_lower = clause_text.lower()
        count = 0
        
        for keyword in self.major_risk_keywords:
            if keyword in text_lower:
                count += 1
        
        return count
    
    def _has_critical_risk_phrases(self, clause_text: str) -> bool:
        """
        Check if clause contains critical risk phrases that force High classification
        """
        text_lower = clause_text.lower()
        return any(phrase in text_lower for phrase in self.critical_risk_phrases)
    
    def _calculate_keyword_boost(self, clause_text: str) -> float:
        """
        Calculate keyword boost based on major asymmetric keywords
        Returns boost value to add to final score
        """
        keyword_count = self._count_major_risk_keywords(clause_text)
        # Each major keyword adds +0.05 boost, max of +0.10
        return min(keyword_count * 0.05, 0.10)
    
    def _dynamic_normalization(self, scores: List[float]) -> float:
        """
        Calculate dynamic normalization factor based on current batch max
        """
        if not scores or max(scores) == 0:
            return 8.5  # fallback to observed max
        
        # Use actual max from current batch, with reasonable bounds
        batch_max = max(scores)
        return max(batch_max, 5.0)  # minimum normalization factor of 5.0
    
    def assess_clause_fast(self, clause_id: str, clause_type: str, clause_text: str) -> Dict[str, Any]:
        """
        Fast risk assessment using rule-based analyzers with refined scoring
        
        Returns:
            Dictionary with risk_level, risk_title, risk_description, problematic_spans, suggestions
        """
        # Step 1: Pre-processing - detect correct clause theme
        detected_clause_type = self._detect_clause_theme(clause_text)
        
        # Use detected type if original is missing or generic
        if not clause_type or clause_type.lower() in ["general", "clause", "section"]:
            clause_type = detected_clause_type
        
        # Analyze financial risk
        financial_risk = self.financial_analyzer.analyze_financial_risk(
            clause_type=clause_type,
            clause_text=clause_text,
            contract_value=None,
            contract_duration=None
        )
        
        # Analyze legal risk
        legal_risk = self.legal_analyzer.analyze_legal_risk(
            clause_type=clause_type,
            clause_text=clause_text,
            governing_law=None,
            industry=None
        )
        
        # Step 2: Apply lower base scores (0.8-1.2 range)
        overall_financial = financial_risk.overall_financial_risk
        overall_legal = legal_risk.overall_legal_risk
        
        # Count actual risk factors detected
        num_financial_factors = len(financial_risk.risk_factors)
        num_legal_factors = len(legal_risk.risk_factors)
        
        # Dynamic normalization using current scores
        current_scores = [overall_financial, overall_legal]
        normalization_factor = self._dynamic_normalization(current_scores)
        
        # Apply lower base scores when no risk factors detected
        if num_financial_factors == 0 and num_legal_factors == 0:
            # Force to very low base scores (0.8-1.2 range)
            normalized_financial = 0.08  # 0.8/10
            normalized_legal = 0.12      # 1.2/10
        else:
            # Normal normalization when risks are detected
            normalized_financial = min(overall_financial / normalization_factor, 1.0)
            normalized_legal = min(overall_legal / normalization_factor, 1.0)
        
        # Step 3: Weighted combination - Legal 70%, Financial 30%
        combined_score = (normalized_financial * 0.30) + (normalized_legal * 0.70)
        
        # Step 4: Apply clause-type multipliers
        clause_multiplier = self.clause_multipliers.get(clause_type, 1.0)
        adjusted_score = min(combined_score * clause_multiplier, 1.0)
        
        # Step 5: Apply keyword boost (+0.05 per major keyword, max +0.10)
        keyword_boost = self._calculate_keyword_boost(clause_text)
        boosted_score = min(adjusted_score + keyword_boost, 1.0)
        
        # Step 6: Count major risk keywords and check critical phrases
        major_keyword_count = self._count_major_risk_keywords(clause_text)
        has_critical_phrases = self._has_critical_risk_phrases(clause_text)
        
        # Step 7: Determine risk level with refined thresholds
        # Hard-coded override rules for critical phrases
        if "unlimited liability" in clause_text.lower() or "broad indemnification" in clause_text.lower():
            risk_level = "High"
        elif has_critical_phrases:
            # Semantic calibration - other critical phrases force High
            risk_level = "High"
        elif boosted_score >= 0.45 and major_keyword_count >= 2:
            risk_level = "High"
        elif boosted_score >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Extract problematic spans
        problematic_spans = []
        all_factors = financial_risk.risk_factors + legal_risk.risk_factors
        
        for factor in all_factors[:3]:  # Top 3 factors
            if "text" in factor:
                problematic_spans.append(factor["text"])
            elif "description" in factor:
                problematic_spans.append(factor["description"])
        
        # Generate risk title and description
        risk_title = self._generate_risk_title(clause_type, boosted_score, all_factors)
        risk_description = self._generate_risk_description(clause_type, financial_risk, legal_risk, all_factors)
        suggestions = self._generate_suggestions(clause_type, risk_level, all_factors)
        
        return {
            "risk_level": risk_level,
            "risk_score": boosted_score,
            "risk_title": risk_title,
            "risk_description": risk_description,
            "problematic_spans": problematic_spans,
            "suggestions": suggestions,
            "financial_risk": overall_financial,
            "legal_risk": overall_legal,
            "clause_type": clause_type,  # Return the corrected clause type
            "major_keywords": major_keyword_count,
            "has_critical_phrases": has_critical_phrases,
            "keyword_boost": keyword_boost,
            "clause_multiplier": clause_multiplier,
            "base_score": combined_score,
            "normalization_factor": normalization_factor
        }
    
    def _generate_risk_title(self, clause_type: str, score: float, factors: List[Dict]) -> str:
        """Generate concise risk title"""
        if score >= 0.7:
            severity = "High"
        elif score >= 0.4:
            severity = "Medium"
        else:
            severity = "Low"
        
        clause_name = clause_type.replace("_", " ").title()
        
        if factors:
            top_factor = factors[0].get("type", "risk")
            return f"{severity} {clause_name} risk: {top_factor.replace('_', ' ').title()}"
        else:
            return f"{severity} {clause_name} risk"
    
    def _generate_risk_description(self, clause_type: str, financial_risk, legal_risk, factors: List[Dict]) -> str:
        """Generate detailed risk description"""
        descriptions = []
        
        if financial_risk.overall_financial_risk >= 0.5:
            descriptions.append(f"Financial risk score: {financial_risk.overall_financial_risk:.2f}")
        
        if legal_risk.overall_legal_risk >= 0.5:
            descriptions.append(f"Legal risk score: {legal_risk.overall_legal_risk:.2f}")
        
        if factors:
            top_factors = factors[:2]
            factor_descs = [f.get("description", f.get("type", "")) for f in top_factors]
            descriptions.extend(factor_descs)
        
        if not descriptions:
            return f"Standard {clause_type.replace('_', ' ')} clause with moderate risk factors."
        
        return ". ".join(descriptions) + "."
    
    def _generate_suggestions(self, clause_type: str, risk_level: str, factors: List[Dict]) -> List[str]:
        """Generate mitigation suggestions"""
        suggestions = []
        
        if risk_level == "High":
            suggestions.append(f"Review {clause_type.replace('_', ' ')} clause carefully with legal counsel")
            suggestions.append("Consider negotiating more favorable terms")
        
        if factors:
            # Add specific suggestions based on risk factors
            for factor in factors[:2]:
                if "suggestion" in factor:
                    suggestions.append(factor["suggestion"])
        
        if not suggestions:
            suggestions.append("Standard clause - monitor for compliance")
        
        return suggestions


def _format_clause_title(clause_type: str) -> str:
    """Format clause type as readable title"""
    if not clause_type:
        return "Unknown Clause"
    return clause_type.replace("_", " ").title()


def _extract_tags(clause_type: str, clause_text: str) -> List[str]:
    """Extract relevant tags for clause"""
    tags = []
    
    if clause_type:
        tags.append(clause_type.replace("_", " ").title())
    
    # Add common legal tags based on keywords
    text_lower = clause_text.lower()
    
    if any(kw in text_lower for kw in ["liability", "liable", "damages"]):
        tags.append("Liability")
    if any(kw in text_lower for kw in ["indemnif", "hold harmless", "defend"]):
        tags.append("Indemnity")
    if any(kw in text_lower for kw in ["terminat", "expire", "end"]):
        tags.append("Termination")
    if any(kw in text_lower for kw in ["payment", "fee", "compensation"]):
        tags.append("Payment")
    if any(kw in text_lower for kw in ["confidential", "proprietary", "non-disclosure"]):
        tags.append("Confidentiality")
    
    return tags[:3]  # Limit to 3 tags


def _generate_summary(processed_doc, clauses_data: List[Dict]) -> Dict[str, Any]:
    """Generate document summary"""
    # Try to extract basic metadata from document
    full_text = processed_doc.full_text[:2000]  # First 2000 chars
    
    # Extract party names (simple heuristic)
    party_names = []
    party_keywords = ["party", "parties", "company", "corporation", "llc", "inc"]
    lines = full_text.split("\n")[:20]  # First 20 lines
    for line in lines:
        if any(kw in line.lower() for kw in party_keywords):
            # Try to extract potential party name
            words = line.split()
            if len(words) > 0:
                potential_name = words[0] if len(words[0]) > 3 else None
                if potential_name and potential_name not in party_names:
                    party_names.append(potential_name)
                    if len(party_names) >= 2:
                        break
    
    # Extract dates (simple pattern)
    import re
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    dates = re.findall(date_pattern, full_text)
    
    effective_date = dates[0] if dates else None
    expiry_date = dates[1] if len(dates) > 1 else None
    
    # Determine contract type from clauses
    clause_types = [c.clause_type or "" for c in processed_doc.clauses if c.clause_type]
    contract_type = "Service Agreement"
    
    if any("employment" in ct.lower() for ct in clause_types):
        contract_type = "Employment Contract"
    elif any("license" in ct.lower() for ct in clause_types):
        contract_type = "License Agreement"
    elif any("nda" in ct.lower() or "confidentiality" in ct.lower() for ct in clause_types):
        contract_type = "Non-Disclosure Agreement"
    
    return {
        "contract_type": contract_type,
        "party_names": party_names[:2] if party_names else ["Party A", "Party B"],
        "effective_date": effective_date,
        "expiry_date": expiry_date
    }


def _calculate_overall_risk_score(clauses_data: List[Dict]) -> int:
    """Calculate overall risk score (0-100)"""
    if not clauses_data:
        return 0
    
    risk_weights = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    
    total_weight = 0
    weighted_sum = 0
    
    for clause in clauses_data:
        risk = clause.get("risk", "Low")
        weight = risk_weights.get(risk, 1)
        weighted_sum += weight
        total_weight += 1
    
    if total_weight == 0:
        return 0
    
    # Normalize to 0-100 scale
    avg_weight = weighted_sum / total_weight
    score = int((avg_weight / 4.0) * 100)  # 4 is max weight
    
    return min(100, max(0, score))

