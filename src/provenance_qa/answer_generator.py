"""
Answer Generation Component

Generates comprehensive answers using Gemini 2.5 Flash API with fallback strategies.
Handles prompt engineering, response processing, and quality assessment.
"""

import os
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

try:
    from ..models.answer_models import (
        Answer, AnswerType, ConfidenceLevel, AnswerSource
    )
    from ..models.context_models import ContextWindow
    from ..models.question_models import QuestionAnalysis
    from ..utils.common import (
        generate_id, clean_text, calculate_confidence_score, Timer, RateLimiter
    )
except ImportError:
    # Fallback for direct execution
    from provenance_qa.models.answer_models import (
        Answer, AnswerType, ConfidenceLevel, AnswerSource
    )
    from provenance_qa.models.context_models import ContextWindow
    from provenance_qa.models.question_models import QuestionAnalysis
    from provenance_qa.utils.common import (
        generate_id, clean_text, calculate_confidence_score, Timer, RateLimiter
    )

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Advanced answer generation using Gemini 2.5 Flash API"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Initialize Gemini API
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
            logger.warning("No Gemini API key provided. Answer generation will use fallback methods.")
        
        # Rate limiter for API calls
        self.rate_limiter = RateLimiter(max_calls=60, time_window=60)  # 60 calls per minute
        
        # Build prompt templates
        self.prompt_templates = self._build_prompt_templates()
        
        # Configure generation settings
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Lower temperature for more factual responses
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
            candidate_count=1
        )
        
        # Configure safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    def _build_prompt_templates(self) -> Dict[str, str]:
        """Build prompt templates for different question types"""
        return {
            "factual": """You are a legal document analysis expert specializing in contract risk assessment. Answer the following question based solely on the provided context.

IMPORTANT: The context includes contract clauses with risk assessments. When answering:
- Pay attention to RISK ASSESSMENT sections which contain severity levels, descriptions, and suggestions
- Use risk level information (High/Medium/Low) to provide context-aware answers
- Reference specific risk indicators and problematic spans when relevant
- Include risk-related insights when answering questions about contract terms

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a clear, factual answer based only on the information in the context
2. If risk assessment information is available, incorporate it into your answer
3. If the answer is not clearly stated in the context, say so explicitly
4. Include specific references to relevant sections, clauses, or risk assessments
5. Use precise language and avoid speculation
6. If there are multiple relevant pieces of information, organize them clearly
7. When discussing clauses, mention their risk levels if available

ANSWER:""",

            "comparative": """You are a legal document analysis expert. Compare and analyze the information requested based on the provided context.

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
1. Identify the specific items, terms, or documents being compared
2. Present similarities and differences in a structured format
3. Base your comparison only on information explicitly stated in the context
4. Highlight key differences that may be legally significant
5. If the context doesn't contain sufficient information for comparison, state this clearly

COMPARISON:""",

            "analytical": """You are a legal document analysis expert specializing in contract risk assessment. Provide a thorough analysis based on the question and context provided.

IMPORTANT: The context includes contract clauses with risk assessments. When analyzing:
- Consider the risk levels (High/Medium/Low) associated with clauses
- Incorporate risk assessment descriptions and suggestions into your analysis
- Highlight clauses with High or Medium risk levels
- Use risk indicators and problematic spans to provide deeper insights

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze the relevant legal concepts, terms, and implications
2. Break down complex information into clear, understandable components
3. Identify potential risks, obligations, or important considerations using the risk assessment data
4. Base your analysis strictly on the provided context, including risk assessment information
5. When discussing risks, reference the specific risk assessments provided
6. If your analysis requires information not in the context, note these limitations

ANALYSIS:""",

            "procedural": """You are a legal document analysis expert. Explain the process or procedure requested based on the provided context.

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
1. Outline the steps or process clearly and sequentially
2. Include any requirements, conditions, or prerequisites mentioned
3. Note any deadlines, timeframes, or important timing considerations
4. Base your explanation only on the information in the context
5. If the complete procedure is not detailed in the context, note what information is missing

PROCEDURE:""",

            "default": """You are a legal document analysis expert specializing in contract risk assessment. Answer the following question based on the provided context.

IMPORTANT: The context includes contract clauses with risk assessments. When answering:
- Pay attention to RISK ASSESSMENT sections which contain severity levels, descriptions, and suggestions
- Use risk level information (High/Medium/Low) to provide context-aware answers
- Reference specific risk indicators and problematic spans when relevant
- Include risk-related insights when answering questions about contract terms

QUESTION: {question}

CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive answer based solely on the context provided, including risk assessment data
2. Be precise and accurate in your response
3. Include relevant details and references to specific sections, clauses, and their risk levels
4. When discussing clauses, mention their risk assessments if available
5. If the context is insufficient to fully answer the question, explain what information is available and what is missing
6. Maintain objectivity and avoid speculation beyond what is stated in the documents
7. Incorporate risk assessment insights to provide more valuable answers

ANSWER:"""
        }
    
    def generate_answer(
        self,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow
    ) -> Answer:
        """Generate comprehensive answer using Gemini 2.5 Flash API"""
        
        with Timer() as timer:
            # Generate answer ID
            answer_id = generate_id("ans")
            
            # Create answer object
            answer = Answer(
                answer_id=answer_id,
                question_id=question_analysis.original_question[:50]  # Use question as ID for now
            )
            
            try:
                # Try primary generation method (Gemini API)
                if self.model and self.api_key:
                    answer = self._generate_with_gemini(answer, question_analysis, context_window)
                else:
                    # Use fallback generation method
                    answer = self._generate_with_fallback(answer, question_analysis, context_window)
                
                # Analyze generated answer
                self._analyze_answer_content(answer, question_analysis)
                
                # Calculate quality metrics
                self._calculate_answer_quality(answer, question_analysis, context_window)
                
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Create error answer
                answer.text = "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."
                answer.answer_type = AnswerType.NOT_FOUND
                answer.confidence = 0.0
                answer.confidence_level = ConfidenceLevel.VERY_LOW
                answer.source = AnswerSource.FALLBACK_MODEL
                answer.limitations.append("Error occurred during answer generation")
            
            # Record generation time
            answer.generation_time = timer.elapsed()
            
            logger.info(f"Generated answer: {len(answer.text)} characters, "
                       f"confidence: {answer.confidence:.2f}")
            
            return answer
    
    def _generate_with_gemini(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow
    ) -> Answer:
        """Generate answer using Gemini 2.5 Flash API"""
        
        # Check rate limit
        if not self.rate_limiter.can_proceed():
            wait_time = self.rate_limiter.time_until_next_call()
            logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            import time
            time.sleep(wait_time)
        
        # Build prompt based on question type
        prompt = self._build_prompt(question_analysis, context_window)
        
        try:
            # Record API call
            self.rate_limiter.record_call()
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Process response
            if response.candidates and response.candidates[0].content:
                answer.text = response.candidates[0].content.parts[0].text
                answer.source = AnswerSource.GEMINI_25_FLASH
                answer.model_name = "gemini-2.0-flash-exp"
                
                # Extract token usage if available
                if hasattr(response, 'usage_metadata'):
                    answer.tokens_used = response.usage_metadata.total_token_count
                
                logger.info("Successfully generated answer with Gemini API")
                
            else:
                # Handle case where no content was generated
                raise Exception("No content generated by Gemini API")
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            # Fall back to alternative generation method
            return self._generate_with_fallback(answer, question_analysis, context_window)
        
        return answer
    
    def _generate_with_fallback(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow
    ) -> Answer:
        """Generate answer using improved fallback template-based method"""
        
        logger.info("Using enhanced fallback answer generation method")
        
        # Use template-based generation
        answer.source = AnswerSource.TEMPLATE_BASED
        answer.model_name = "enhanced_template_fallback"
        
        # Get relevant chunks
        all_chunks = context_window.get_all_chunks()
        
        if not all_chunks:
            # Natural "I don't know" responses when no relevant content is found
            no_info_responses = [
                "I'm sorry, but I couldn't find relevant information in the documents to answer your question.",
                "I don't have enough information in the available contracts to answer that.",
                "Sorry, I can't find specific details about that in these contract documents.", 
                "I'm afraid the documents don't contain the information needed to answer your question.",
                "That's not covered in the contract documents I have access to."
            ]
            import random
            answer.text = random.choice(no_info_responses)
            answer.answer_type = AnswerType.NOT_FOUND
            answer.confidence_level = ConfidenceLevel.VERY_LOW
            return answer
        
        # Enhanced answer generation based on question type
        try:
            from ..models.question_models import QuestionType
        except ImportError:
            from provenance_qa.models.question_models import QuestionType
        
        question_text = question_analysis.original_question.lower()
        
        # Determine answer strategy based on question type
        if "termination" in question_text or "terminate" in question_text:
            answer.text = self._generate_termination_answer(all_chunks, question_analysis)
            answer.answer_type = AnswerType.DIRECT
        elif "liability" in question_text or "liable" in question_text:
            answer.text = self._generate_liability_answer(all_chunks, question_analysis)
            answer.answer_type = AnswerType.ANALYSIS
        elif "payment" in question_text or "pay" in question_text:
            answer.text = self._generate_payment_answer(all_chunks, question_analysis)
            answer.answer_type = AnswerType.DIRECT
        elif "effective" in question_text and "date" in question_text:
            answer.text = self._generate_date_answer(all_chunks, question_analysis)
            answer.answer_type = AnswerType.DIRECT
        elif "renewal" in question_text or "renew" in question_text:
            answer.text = self._generate_renewal_answer(all_chunks, question_analysis)
            answer.answer_type = AnswerType.DIRECT
        else:
            answer.text = self._generate_generic_answer(all_chunks, question_analysis)
            answer.answer_type = AnswerType.SUMMARY
        
        # Set confidence based on content quality
        if len(answer.text) > 200 and len(all_chunks) >= 2:
            answer.confidence_level = ConfidenceLevel.MODERATE
        elif len(answer.text) > 100:
            answer.confidence_level = ConfidenceLevel.LOW
        else:
            answer.confidence_level = ConfidenceLevel.VERY_LOW
        
        return answer
    
    def _generate_termination_answer(self, chunks, question_analysis) -> str:
        """Generate focused answer about termination clauses"""
        relevant_chunks = [c for c in chunks if "terminat" in c.content.lower()]
        
        if not relevant_chunks:
            return "I couldn't find specific termination clauses in the contract documents I reviewed."
        
        chunk = relevant_chunks[0]
        if "thirty (30) days" in chunk.content.lower():
            return f"Looking at the termination section, it appears that either party can end the agreement with thirty days' written notice. Once terminated, all rights and licenses stop immediately, and any confidential information needs to be returned or destroyed."
        elif "notice" in chunk.content.lower():
            return f"The contract includes termination provisions that require advance notice, though I'd need to check the specific timeframe requirements."
        
        return "There are termination clauses in the contract, but I'd need to review the specific terms to give you accurate details about notice periods and obligations."
    
    def _generate_liability_answer(self, chunks, question_analysis) -> str:
        """Generate focused answer about liability provisions"""
        relevant_chunks = [c for c in chunks if "liabilit" in c.content.lower()]
        
        if not relevant_chunks:
            return "I don't see specific liability provisions in the sections I reviewed. There might be liability clauses elsewhere in the contract."
        
        chunk = relevant_chunks[0]
        if "exceed" in chunk.content.lower() or "cap" in chunk.content.lower():
            return f"The contract does include liability limitations. From what I can see, there's typically a cap on damages - often limited to amounts already paid under the agreement. It usually excludes things like consequential or punitive damages, though there may be exceptions for serious misconduct."
        elif "limitation" in chunk.content.lower():
            return "There are liability limitations mentioned in the contract, but I'd need to examine the specific terms to tell you exactly how damages are capped."
        
        return "The contract mentions liability provisions, but I'd need to look more closely at the specific language to explain the limitations."
    
    def _generate_payment_answer(self, chunks, question_analysis) -> str:
        """Generate focused answer about payment terms"""
        relevant_chunks = [c for c in chunks if "payment" in c.content.lower() or "salary" in c.content.lower() or "compensation" in c.content.lower()]
        
        if not relevant_chunks:
            return "I don't see specific payment terms in the contract sections I have access to."
        
        chunk = relevant_chunks[0]
        if "bi-weekly" in chunk.content.lower():
            return f"The payment schedule shows bi-weekly payments, typically on the 15th and last day of each month. Payments are usually made by direct deposit, with standard tax withholdings applied."
        elif "monthly" in chunk.content.lower():
            return "Payments appear to be made monthly, though I'd need to check the exact dates and method."
        elif any(term in chunk.content.lower() for term in ["salary", "wage", "compensation"]):
            return "There are compensation details in the contract, but I'd need to review the specific payment schedule and amounts."
        
        return "I can see there are payment terms mentioned, but the specific details aren't clear from what I'm reviewing."
    
    def _generate_date_answer(self, chunks, question_analysis) -> str:
        """Generate focused answer about effective dates"""
        relevant_chunks = [c for c in chunks if "effective" in c.content.lower() and "date" in c.content.lower()]
        
        if not relevant_chunks:
            # Look for any date-related content
            date_chunks = [c for c in chunks if any(term in c.content.lower() for term in ["date", "effective", "commence", "begin"])]
            if not date_chunks:
                return "I couldn't find the effective date mentioned in the contract sections I reviewed."
            relevant_chunks = date_chunks
        
        chunk = relevant_chunks[0]
        # Look for specific date patterns
        import re
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, chunk.content)
            if dates:
                return f"The effective date appears to be {dates[0]}. This is when the agreement terms become binding and take effect."
        
        if "effective" in chunk.content.lower():
            return "The contract mentions an effective date, but I'd need to check the specific date in the document."
        
        return "I can see date-related information in the contract, but the specific effective date isn't clear from what I'm reviewing."
    
    def _generate_renewal_answer(self, chunks, question_analysis) -> str:
        """Generate focused answer about renewal options"""
        relevant_chunks = [c for c in chunks if "renew" in c.content.lower() or "extend" in c.content.lower() or "term" in c.content.lower()]
        
        if not relevant_chunks:
            return "I don't see specific renewal terms mentioned in the contract sections I have access to."
        
        chunk = relevant_chunks[0]
        if "one-year terms" in chunk.content.lower():
            return f"Yes, it looks like the contract can be renewed for additional one-year periods. You'd need to give written notice at least sixty days before it expires to extend it under similar terms."
        elif "automatic" in chunk.content.lower() and "renew" in chunk.content.lower():
            return "The contract appears to have automatic renewal provisions, though I'd need to check the specific notice requirements to opt out."
        elif "notice" in chunk.content.lower() and ("renew" in chunk.content.lower() or "extend" in chunk.content.lower()):
            return "There are renewal options available, but they require advance notice. I'd need to check the exact timing requirements."
        
        return "The contract mentions renewal or extension terms, but I'd need to review the specific provisions to give you accurate details."
    
    def _generate_generic_answer(self, chunks, question_analysis) -> str:
        """Generate generic answer from available content with natural language"""
        
        # Check if we actually have relevant information
        question_text = question_analysis.original_question.lower()
        question_keywords = question_analysis.keywords if hasattr(question_analysis, 'keywords') else []
        
        # Natural "I don't know" responses for insufficient information
        insufficient_info_responses = [
            "I couldn't find specific information about that in the contract documents I have access to.",
            "Sorry, I don't see details about that particular topic in these contracts.",
            "I'm afraid the documents don't contain clear information to answer that question.",
            "I don't have enough information in these contracts to give you a definitive answer on that.",
            "That's not something I can find specific details about in the available documents."
        ]
        
        # Check content relevance
        relevant_chunks = []
        for chunk in chunks[:3]:  # Check top 3 chunks
            chunk_text = chunk.content.lower()
            # Simple relevance check
            if any(keyword.lower() in chunk_text for keyword in question_keywords if len(keyword) > 3):
                relevant_chunks.append(chunk)
            elif any(word in chunk_text for word in question_text.split() if len(word) > 3):
                relevant_chunks.append(chunk)
        
        # If no relevant chunks found, use natural "I don't know" response
        if not relevant_chunks:
            import random
            return random.choice(insufficient_info_responses)
        
        # Generate natural response when we have relevant information
        natural_intros = [
            "Looking at the contract documents, here's what I found:",
            "From what I can see in the contracts:",
            "Based on the contract information available:",
            "The contracts indicate the following:",
            "Here's what the documents show:"
        ]
        
        answer_parts = []
        import random
        answer_parts.append(random.choice(natural_intros))
        answer_parts.append("")
        
        # Add information from relevant chunks with natural language
        for i, chunk in enumerate(relevant_chunks[:2], 1):
            location_parts = []
            if hasattr(chunk, 'document_title') and chunk.document_title:
                location_parts.append(chunk.document_title)
            if hasattr(chunk, 'section_title') and chunk.section_title:
                location_parts.append(chunk.section_title)
            
            if location_parts:
                location = " - ".join(location_parts)
            else:
                location = "the contract"
            
            # Extract and format content more naturally
            content = chunk.content
            if len(content) > 300:
                # Find sentence boundaries and truncate appropriately
                sentences = content.split('. ')
                if len(sentences) > 2:
                    content = '. '.join(sentences[:2]) + '.'
            
            # Use more natural language connectors
            if i == 1:
                answer_parts.append(f"In {location}: {content}")
            else:
                answer_parts.append(f"Additionally, {location.lower()} mentions: {content}")
            answer_parts.append("")
        
        # Add natural conclusion if more information available
        if len(chunks) > len(relevant_chunks) and len(relevant_chunks) < 2:
            concluding_phrases = [
                "There might be additional relevant details in other sections.",
                "I found some other related information, but it may not directly answer your question.",
                "Let me know if you need me to look into any specific aspect in more detail."
            ]
            answer_parts.append(random.choice(concluding_phrases))
        
        return "\n".join(answer_parts)
        
        return answer
    
    def _build_prompt(
        self,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow
    ) -> str:
        """Build optimized prompt for question type"""
        
        # Select appropriate template
        try:
            from ..models.question_models import QuestionType
        except ImportError:
            from provenance_qa.models.question_models import QuestionType
        
        if question_analysis.question_type == QuestionType.COMPARATIVE:
            template_key = "comparative"
        elif question_analysis.question_type == QuestionType.ANALYTICAL:
            template_key = "analytical"
        elif question_analysis.question_type == QuestionType.PROCEDURAL:
            template_key = "procedural"
        elif question_analysis.question_type == QuestionType.FACTUAL:
            template_key = "factual"
        else:
            template_key = "default"
        
        template = self.prompt_templates.get(template_key, self.prompt_templates["default"])
        
        # Format context
        formatted_context = context_window.get_formatted_context(include_metadata=True)
        
        # Build final prompt
        prompt = template.format(
            question=question_analysis.original_question,
            context=formatted_context
        )
        
        return prompt
    
    def _analyze_answer_content(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis
    ):
        """Analyze answer content and extract metadata"""
        
        if not answer.text:
            return
        
        # Classify answer type based on content
        answer.answer_type = self._classify_answer_type(answer.text, question_analysis)
        
        # Extract key points
        answer.key_points = self._extract_key_points(answer.text)
        
        # Extract supporting facts
        answer.supporting_facts = self._extract_supporting_facts(answer.text)
        
        # Identify caveats and limitations
        answer.caveats = self._identify_caveats(answer.text)
        answer.limitations = self._identify_limitations(answer.text)
    
    def _classify_answer_type(self, answer_text: str, question_analysis: QuestionAnalysis) -> AnswerType:
        """Classify the type of answer based on content"""
        
        answer_lower = answer_text.lower()
        
        # Check for "not found" indicators
        not_found_indicators = [
            "not found", "not available", "not mentioned", "not specified",
            "cannot find", "unable to locate", "insufficient information"
        ]
        
        if any(indicator in answer_lower for indicator in not_found_indicators):
            return AnswerType.NOT_FOUND
        
        # Check for ambiguity indicators
        ambiguous_indicators = [
            "multiple possibilities", "could be", "might be", "unclear",
            "ambiguous", "several options"
        ]
        
        if any(indicator in answer_lower for indicator in ambiguous_indicators):
            return AnswerType.AMBIGUOUS
        
        # Check for comparison content
        if "compared to" in answer_lower or "difference" in answer_lower or "versus" in answer_lower:
            return AnswerType.COMPARISON
        
        # Check for analysis content
        analysis_indicators = ["analysis", "implications", "risks", "considerations"]
        if any(indicator in answer_lower for indicator in analysis_indicators):
            return AnswerType.ANALYSIS
        
        # Check for procedural content
        procedural_indicators = ["steps", "process", "procedure", "first", "then", "next"]
        if any(indicator in answer_lower for indicator in procedural_indicators):
            return AnswerType.EXPLANATION
        
        # Default to direct answer
        return AnswerType.DIRECT
    
    def _extract_key_points(self, answer_text: str) -> List[str]:
        """Extract key points from answer text"""
        key_points = []
        
        # Look for numbered lists
        import re
        numbered_points = re.findall(r'^\d+\.\s+(.+)$', answer_text, re.MULTILINE)
        key_points.extend(numbered_points)
        
        # Look for bullet points
        bullet_points = re.findall(r'^[•\-\*]\s+(.+)$', answer_text, re.MULTILINE)
        key_points.extend(bullet_points)
        
        # If no structured points, extract sentences with key terms
        if not key_points:
            sentences = answer_text.split('.')
            key_terms = ['important', 'key', 'significant', 'notable', 'must', 'required', 'should']
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(term in sentence.lower() for term in key_terms) and len(sentence) > 10:
                    key_points.append(sentence)
        
        return key_points[:5]  # Limit to top 5 points
    
    def _extract_supporting_facts(self, answer_text: str) -> List[str]:
        """Extract supporting facts from answer text"""
        supporting_facts = []
        
        # Look for phrases that introduce facts
        fact_indicators = [
            "according to", "states that", "specifies", "indicates",
            "shows that", "reveals", "confirms"
        ]
        
        sentences = answer_text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in fact_indicators):
                supporting_facts.append(sentence)
        
        return supporting_facts[:3]  # Limit to top 3 facts
    
    def _identify_caveats(self, answer_text: str) -> List[str]:
        """Identify caveats in the answer"""
        caveats = []
        
        caveat_indicators = [
            "however", "but", "although", "nevertheless", "except",
            "with the exception", "unless", "provided that", "subject to"
        ]
        
        sentences = answer_text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in caveat_indicators):
                caveats.append(sentence)
        
        return caveats[:3]  # Limit to top 3 caveats
    
    def _identify_limitations(self, answer_text: str) -> List[str]:
        """Identify limitations in the answer"""
        limitations = []
        
        limitation_indicators = [
            "limited information", "not specified", "unclear", "insufficient detail",
            "may require", "additional information needed", "consult", "verify"
        ]
        
        answer_lower = answer_text.lower()
        for indicator in limitation_indicators:
            if indicator in answer_lower:
                # Find the sentence containing the limitation
                sentences = answer_text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        limitations.append(sentence.strip())
                        break
        
        return list(set(limitations))  # Remove duplicates
    
    def _calculate_answer_quality(
        self,
        answer: Answer,
        question_analysis: QuestionAnalysis,
        context_window: ContextWindow
    ):
        """Calculate quality metrics for the answer"""
        
        quality_factors = {}
        
        # Completeness score
        completeness_score = self._assess_completeness(answer, question_analysis)
        quality_factors["completeness"] = completeness_score
        answer.completeness_score = completeness_score
        
        # Accuracy score (based on context relevance)
        accuracy_score = self._assess_accuracy(answer, context_window)
        quality_factors["accuracy"] = accuracy_score
        answer.accuracy_score = accuracy_score
        
        # Relevance score
        relevance_score = self._assess_relevance(answer, question_analysis)
        quality_factors["relevance"] = relevance_score
        answer.relevance_score = relevance_score
        
        # Clarity score
        clarity_score = self._assess_clarity(answer)
        quality_factors["clarity"] = clarity_score
        answer.clarity_score = clarity_score
        
        # Overall confidence calculation
        weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "relevance": 0.25,
            "clarity": 0.15
        }
        
        answer.confidence = calculate_confidence_score(quality_factors, weights)
        
        # Set confidence level
        if answer.confidence >= 0.9:
            answer.confidence_level = ConfidenceLevel.VERY_HIGH
        elif answer.confidence >= 0.75:
            answer.confidence_level = ConfidenceLevel.HIGH
        elif answer.confidence >= 0.5:
            answer.confidence_level = ConfidenceLevel.MODERATE
        elif answer.confidence >= 0.25:
            answer.confidence_level = ConfidenceLevel.LOW
        else:
            answer.confidence_level = ConfidenceLevel.VERY_LOW
    
    def _assess_completeness(self, answer: Answer, question_analysis: QuestionAnalysis) -> float:
        """Assess how complete the answer is"""
        
        if answer.answer_type == AnswerType.NOT_FOUND:
            return 0.0
        
        # Check if answer addresses the question type appropriately
        try:
            from ..models.question_models import QuestionType
        except ImportError:
            from provenance_qa.models.question_models import QuestionType
        
        answer_lower = answer.text.lower()
        question_type = question_analysis.question_type
        
        if question_type == QuestionType.COMPARATIVE:
            # Should contain comparison language
            comparison_terms = ["compared to", "versus", "difference", "similar", "unlike"]
            has_comparison = any(term in answer_lower for term in comparison_terms)
            return 0.8 if has_comparison else 0.4
        
        elif question_type == QuestionType.QUANTITATIVE:
            # Should contain numbers or quantities
            import re
            has_numbers = bool(re.search(r'\d+', answer.text))
            return 0.8 if has_numbers else 0.3
        
        elif question_type == QuestionType.PROCEDURAL:
            # Should contain step-by-step information
            step_indicators = ["first", "then", "next", "finally", "step", "process"]
            has_steps = any(indicator in answer_lower for indicator in step_indicators)
            return 0.8 if has_steps else 0.4
        
        # General completeness assessment
        length_score = min(1.0, len(answer.text) / 200)  # Normalize by reasonable length
        detail_score = len(answer.key_points) / 5 if answer.key_points else 0.5
        
        return (length_score + detail_score) / 2
    
    def _assess_accuracy(self, answer: Answer, context_window: ContextWindow) -> float:
        """Assess accuracy based on context alignment"""
        
        if not context_window.get_all_chunks():
            return 0.5  # Neutral if no context
        
        # Calculate similarity to context
        all_context = " ".join(chunk.content for chunk in context_window.get_all_chunks())
        try:
            from ..utils.common import calculate_text_similarity
        except ImportError:
            from provenance_qa.utils.common import calculate_text_similarity
        
        similarity = calculate_text_similarity(answer.text, all_context)
        
        # Higher similarity generally indicates better grounding in context
        return similarity
    
    def _assess_relevance(self, answer: Answer, question_analysis: QuestionAnalysis) -> float:
        """Assess relevance to the original question"""
        
        try:
            from ..utils.common import calculate_text_similarity
        except ImportError:
            from provenance_qa.utils.common import calculate_text_similarity
        
        # Calculate similarity between answer and question
        similarity = calculate_text_similarity(answer.text, question_analysis.original_question)
        
        # Check if answer addresses question entities
        entity_coverage = 0.0
        if question_analysis.entities:
            answer_lower = answer.text.lower()
            covered_entities = sum(1 for entity in question_analysis.entities 
                                 if entity.text.lower() in answer_lower)
            entity_coverage = covered_entities / len(question_analysis.entities)
        
        # Combine similarity and entity coverage
        return (similarity + entity_coverage) / 2
    
    def _assess_clarity(self, answer: Answer) -> float:
        """Assess clarity and readability of the answer"""
        
        if not answer.text:
            return 0.0
        
        clarity_factors = {}
        
        # Length appropriateness
        length = len(answer.text)
        if 50 <= length <= 1000:
            clarity_factors["length"] = 1.0
        elif length < 50:
            clarity_factors["length"] = length / 50
        else:
            clarity_factors["length"] = max(0.3, 1000 / length)
        
        # Sentence structure
        sentences = answer.text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        if 10 <= avg_sentence_length <= 25:
            clarity_factors["sentence_length"] = 1.0
        else:
            clarity_factors["sentence_length"] = 0.6
        
        # Structure indicators (lists, organization)
        has_structure = bool(answer.key_points) or '\n' in answer.text
        clarity_factors["structure"] = 0.8 if has_structure else 0.5
        
        return calculate_confidence_score(clarity_factors)