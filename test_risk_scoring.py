#!/usr/bin/env python3
"""
Test script for updated risk scoring logic in fast_analysis.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fast_analysis import RiskAnalyzerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=['Low', 'Medium', 'High'], average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=['Low', 'Medium', 'High'])
    
    return {
        'accuracy': accuracy,
        'per_class': {
            'Low': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': support[0]},
            'Medium': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': support[1]},
            'High': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2], 'support': support[2]}
        },
        'macro_avg': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'weighted_avg': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1},
        'confusion_matrix': conf_matrix
    }

def print_detailed_metrics(metrics):
    """Print detailed metrics in a readable format"""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVALUATION METRICS")
    print(f"{'='*60}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"   Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    
    print(f"\nPER-CLASS METRICS:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 55)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:<10} {class_metrics['precision']:<10.3f} {class_metrics['recall']:<10.3f} "
              f"{class_metrics['f1']:<10.3f} {int(class_metrics['support']):<10}")
    
    print(f"\nAVERAGE METRICS:")
    print(f"   Macro Avg    - Precision: {metrics['macro_avg']['precision']:.3f}, "
          f"Recall: {metrics['macro_avg']['recall']:.3f}, F1: {metrics['macro_avg']['f1']:.3f}")
    print(f"   Weighted Avg - Precision: {metrics['weighted_avg']['precision']:.3f}, "
          f"Recall: {metrics['weighted_avg']['recall']:.3f}, F1: {metrics['weighted_avg']['f1']:.3f}")
    
    print(f"\nCONFUSION MATRIX:")
    print("      Predicted")
    print("      Low  Med  High")
    labels = ['Low', 'Med', 'High']
    for i, (actual_label, row) in enumerate(zip(['Low', 'Medium', 'High'], metrics['confusion_matrix'])):
        if i == 0:
            print(f"A Low   {row[0]:3d}  {row[1]:3d}   {row[2]:3d}")
        elif i == 1:
            print(f"c Med   {row[0]:3d}  {row[1]:3d}   {row[2]:3d}")
        else:
            print(f"t High  {row[0]:3d}  {row[1]:3d}   {row[2]:3d}")
    
    # Risk analysis insights
    print(f"\nKEY INSIGHTS:")
    
    # High-risk detection performance
    high_metrics = metrics['per_class']['High']
    if high_metrics['recall'] < 0.8:
        print(f"   High-risk recall is low ({high_metrics['recall']:.3f}) - missing {(1-high_metrics['recall'])*100:.0f}% of high-risk clauses")
    else:
        print(f"   Good high-risk detection - catching {high_metrics['recall']*100:.0f}% of high-risk clauses")
    
    if high_metrics['precision'] < 0.7:
        print(f"   High-risk precision is low ({high_metrics['precision']:.3f}) - {(1-high_metrics['precision'])*100:.0f}% false positives")
    else:
        print(f"   Good high-risk precision - {high_metrics['precision']*100:.0f}% of high predictions are correct")
    
    # Overall balance
    f1_scores = [metrics['per_class'][cls]['f1'] for cls in ['Low', 'Medium', 'High']]
    f1_std = np.std(f1_scores)
    if f1_std > 0.2:
        print(f"   Unbalanced performance across risk levels (F1 std: {f1_std:.3f})")
    else:
        print(f"   Balanced performance across risk levels (F1 std: {f1_std:.3f})")

def test_risk_scoring():
    """Test the updated risk scoring logic with sample clauses"""
    
    analyzer = RiskAnalyzerFast()
    
    # Test cases with different risk levels
    test_clauses = [
        {
            "id": "1",
            "type": "liability", 
            "text": "Party A shall have unlimited liability for any damages, without cause termination, and sole discretion over all matters. This is irrevocable and absolute.",
            "expected": "High",
            "description": "Multiple major keywords + unlimited liability"
        },
        {
            "id": "2", 
            "type": "payment",
            "text": "Payment shall be made within 30 days of invoice. Late fees may apply at 1.5% per month.",
            "expected": "Low",
            "description": "Standard payment terms"
        },
        {
            "id": "3",
            "type": "termination", 
            "text": "Either party may terminate this agreement with 30 days written notice for any reason.",
            "expected": "Low",
            "description": "Standard termination clause"
        },
        {
            "id": "4",
            "type": "indemnity",
            "text": "Party B shall provide broad indemnification and hold harmless Party A from all claims, including unlimited liability and unconditional waiver of rights.",
            "expected": "High",
            "description": "Critical phrases: broad indemnification + unlimited liability"
        },
        {
            "id": "5",
            "type": "confidentiality",
            "text": "Both parties agree to maintain confidentiality of proprietary information for a period of 5 years.",
            "expected": "Low",
            "description": "Standard confidentiality clause"
        },
        {
            "id": "6",
            "type": "arbitration",
            "text": "All disputes shall be resolved through binding arbitration with exclusive jurisdiction and no reciprocal rights.",
            "expected": "High", 
            "description": "Binding arbitration + exclusive + no reciprocal"
        },
        {
            "id": "7",
            "type": "assignment",
            "text": "This agreement may be assigned by either party with written consent of the other party.",
            "expected": "Low",
            "description": "Standard assignment clause"
        },
        {
            "id": "8",
            "type": "scope",
            "text": "The scope of work includes basic consulting services as defined in Exhibit A.",
            "expected": "Low",
            "description": "Basic scope definition"
        }
    ]

    print("Testing Updated Risk Scoring Logic")
    print("=" * 50)
    
    for clause in test_clauses:
        result = analyzer.assess_clause_fast(
            clause_id=clause["id"],
            clause_type=clause["type"], 
            clause_text=clause["text"]
        )
        
        print(f"\nClause {clause['id']} ({clause['type']}):")
        print(f"  Description: {clause['description']}")
        print(f"  Expected: {clause['expected']}")
        print(f"  Actual: {result['risk_level']}")
        print(f"  Final Score: {result['risk_score']:.3f}")
        print(f"  Base Score: {result['base_score']:.3f}")
        print(f"  Keyword Boost: +{result['keyword_boost']:.3f}")
        print(f"  Clause Multiplier: {result['clause_multiplier']}x")
        print(f"  Major Keywords: {result['major_keywords']}")
        print(f"  Critical Phrases: {result['has_critical_phrases']}")
        print(f"  Corrected Type: {result['clause_type']}")
        print(f"  Match: {'✓' if result['risk_level'] == clause['expected'] else '✗'}")
        
        if result['risk_level'] != clause['expected']:
            print(f" MISMATCH - Expected {clause['expected']}, got {result['risk_level']}")
    
    # Collect predictions for comprehensive metrics
    y_true = [clause['expected'] for clause in test_clauses]
    y_pred = []
    
    for clause in test_clauses:
        result = analyzer.assess_clause_fast(
            clause_id=clause["id"],
            clause_type=clause["type"], 
            clause_text=clause["text"]
        )
        y_pred.append(result['risk_level'])
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    
    # Print detailed analysis
    print_detailed_metrics(metrics)
    
    # Summary statistics
    total_tests = len(test_clauses)
    matches = sum(1 for i, clause in enumerate(test_clauses) if y_pred[i] == clause['expected'])
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {matches}/{total_tests} correct predictions ({matches/total_tests*100:.1f}% accuracy)")
    
    # Recommendations based on metrics
    if metrics['accuracy'] < 0.7:
        print("RECOMMENDATION: Model needs significant improvement")
    elif metrics['accuracy'] < 0.85:
        print("RECOMMENDATION: Model performance is acceptable but could be improved")
    else:
        print("RECOMMENDATION: Model performance is good")
    
    print("Risk scoring evaluation completed!")
    return metrics

if __name__ == "__main__":
    test_risk_scoring()