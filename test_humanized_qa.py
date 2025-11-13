#!/usr/bin/env python3
"""
Test script for humanized QA responses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.provenance_qa import create_qa_engine

def test_humanized_responses():
    """Test that QA responses are more natural and human-like"""
    
    # Create QA engine
    qa_engine = create_qa_engine()
    
    # Test questions that might get "I don't know" responses
    test_questions = [
        "What is the governing law?",
        "What are the termination clauses?", 
        "What are the payment terms?",
        "When does this contract expire?",
        "What is the liability cap?",
        "Can this contract be renewed?",
        "What happens if there's a breach?",
        "Who are the parties to this agreement?"
    ]
    
    print("Testing Humanized QA Responses")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        try:
            # Ask the question
            response = qa_engine.ask_question(question)
            
            # Print the response
            print(f"   Answer: {response.answer[:200]}...")
            print(f"   Type: {response.answer_type if hasattr(response, 'answer_type') else 'N/A'}")
            print(f"   Confidence: {response.overall_confidence:.2f}")
            
            # Check if response is more natural
            robotic_indicators = [
                "Based on the available contract documents:",
                "According to the",
                "Document Summary:",
                "Additional relevant information is available in"
            ]
            
            is_robotic = any(indicator in response.answer for indicator in robotic_indicators)
            natural_indicators = [
                "I couldn't find", "I don't see", "I'm sorry", "I'm afraid",
                "Looking at", "From what I can see", "It appears", "It looks like"
            ]
            
            is_natural = any(indicator in response.answer for indicator in natural_indicators)
            
            print(f"   Assessment: {'✓ Natural' if is_natural else '✗ Robotic' if is_robotic else '- Neutral'}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Humanized response test completed!")

if __name__ == "__main__":
    test_humanized_responses()