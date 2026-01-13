"""
SLM2 Benchmark Evaluation Script

Evaluates SLM2 (knowledge-graph verbalizer) outputs against test dataset.
Computes accuracy metrics broken down by hop count and label type.

Evaluation Contract:
- For label=false: Only "I don't know." is correct
- For label=true: Must verbalize KG facts faithfully
"""

import json
import re
from typing import Dict, List, Tuple
from pathlib import Path


class SLM2Evaluator:
    """Evaluates SLM2 predictions against gold outputs."""
    
    def __init__(self, output_file: str):
        """
        Initialize evaluator with predictions.
        
        Args:
            output_file: Path to JSONL file containing predictions
        """
        self.output_file = Path(output_file)
        self.predictions = []
        self.results = {
            'correct': 0,
            'total': 0,
            'hop_1': {'correct': 0, 'total': 0},
            'hop_2': {'correct': 0, 'total': 0},
            'label_true': {'correct': 0, 'total': 0},
            'label_false': {'correct': 0, 'total': 0},
            'errors': {
                'hallucination': [],
                'over_verbalization': [],
                'under_verbalization': [],
                'wrong_decision': [],
                'false_confidence': [],
                'multi_hop_failure': [],
                'other': []
            }
        }
        
    def normalize_text(self, text: str) -> str:
        """
        Apply minimal normalization to text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        # Trim whitespace
        text = text.strip()
        
        # Normalize quotes (smart quotes to ASCII)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def is_exact_idk(self, text: str) -> bool:
        """
        Check if text is exactly "I don't know."
        
        Args:
            text: Text to check
            
        Returns:
            True if exact match, False otherwise
        """
        normalized = self.normalize_text(text)
        return normalized == "I don't know."
    
    def has_placeholder(self, text: str) -> bool:
        """
        Check if text contains placeholder values.
        
        Args:
            text: Text to check
            
        Returns:
            True if contains placeholder, False otherwise
        """
        placeholders = ['SOME_VALUE', 'INTERMEDIATE', 'probably']
        return any(ph in text for ph in placeholders)
    
    def categorize_error(self, example: Dict) -> str:
        """
        Categorize the type of error made.
        
        Args:
            example: Test example with prediction
            
        Returns:
            Error category string
        """
        predicted = example['predicted_output']
        gold = example['gold_output']
        label = example['label'][0]
        hop_count = example['hop_count']
        
        # For label=false examples
        if not label:
            if self.has_placeholder(predicted):
                return 'hallucination'
            elif not self.is_exact_idk(predicted):
                return 'false_confidence'
            else:
                return 'other'
        
        # For label=true examples
        else:
            if self.is_exact_idk(predicted):
                if hop_count == 2:
                    return 'multi_hop_failure'
                else:
                    return 'wrong_decision'
            elif self.has_placeholder(predicted):
                return 'under_verbalization'
            elif len(predicted) > len(gold) * 1.5:  # Heuristic for extra info
                return 'over_verbalization'
            else:
                return 'other'
    
    def evaluate_example(self, example: Dict) -> bool:
        """
        Evaluate a single example.
        
        Args:
            example: Test example with gold and predicted outputs
            
        Returns:
            True if correct, False otherwise
        """
        predicted = self.normalize_text(example['predicted_output'])
        gold = self.normalize_text(example['gold_output'])
        label = example['label'][0]
        
        # For label=false: Only exact "I don't know." is correct
        if not label:
            return self.is_exact_idk(predicted)
        
        # For label=true: Check for faithful verbalization
        else:
            # Wrong if says "I don't know."
            if self.is_exact_idk(predicted):
                return False
            
            # Wrong if contains placeholders
            if self.has_placeholder(predicted):
                return False
            
            # For exact match comparison (can be relaxed to semantic similarity)
            # Using normalized exact match for now
            return predicted == gold
    
    def load_predictions(self):
        """Load predictions from JSONL file."""
        print(f"Loading predictions from {self.output_file}...")
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.predictions.append(json.loads(line))
        print(f"Loaded {len(self.predictions)} predictions")
    
    def run_evaluation(self):
        """Run evaluation on all predictions."""
        print("\nEvaluating predictions...")
        
        for example in self.predictions:
            is_correct = self.evaluate_example(example)
            hop_count = example['hop_count']
            label = example['label'][0]
            
            # Update overall counts
            self.results['total'] += 1
            if is_correct:
                self.results['correct'] += 1
            
            # Update hop-specific counts
            hop_key = f'hop_{hop_count}'
            if hop_key in self.results:
                self.results[hop_key]['total'] += 1
                if is_correct:
                    self.results[hop_key]['correct'] += 1
            
            # Update label-specific counts
            label_key = 'label_true' if label else 'label_false'
            self.results[label_key]['total'] += 1
            if is_correct:
                self.results[label_key]['correct'] += 1
            
            # Categorize errors
            if not is_correct:
                error_type = self.categorize_error(example)
                self.results['errors'][error_type].append({
                    'id': example['id'],
                    'input': example['input'],
                    'predicted': example['predicted_output'],
                    'gold': example['gold_output'],
                    'label': label,
                    'hop_count': hop_count
                })
    
    def compute_accuracy(self, correct: int, total: int) -> float:
        """Compute accuracy percentage."""
        return (correct / total * 100) if total > 0 else 0.0
    
    def print_results(self):
        """Print evaluation results in a formatted table."""
        print("\n" + "="*70)
        print(" SLM2 BENCHMARK EVALUATION RESULTS")
        print("="*70)
        
        # Main metrics table
        print("\n📊 ACCURACY METRICS")
        print("-" * 70)
        print(f"{'Metric':<30} {'Correct':<12} {'Total':<12} {'Accuracy':<15}")
        print("-" * 70)
        
        # Overall accuracy
        overall_acc = self.compute_accuracy(
            self.results['correct'], 
            self.results['total']
        )
        print(f"{'Overall Accuracy':<30} "
              f"{self.results['correct']:<12} "
              f"{self.results['total']:<12} "
              f"{overall_acc:>6.2f}%")
        
        # Single-hop accuracy (1-hop)
        hop1_acc = self.compute_accuracy(
            self.results['hop_1']['correct'],
            self.results['hop_1']['total']
        )
        print(f"{'Single-Hop Accuracy (1-hop)':<30} "
              f"{self.results['hop_1']['correct']:<12} "
              f"{self.results['hop_1']['total']:<12} "
              f"{hop1_acc:>6.2f}%")
        
        # 2-hop accuracy
        hop2_acc = self.compute_accuracy(
            self.results['hop_2']['correct'],
            self.results['hop_2']['total']
        )
        print(f"{'2-Hop Accuracy':<30} "
              f"{self.results['hop_2']['correct']:<12} "
              f"{self.results['hop_2']['total']:<12} "
              f"{hop2_acc:>6.2f}%")
        
        print("-" * 70)
        
        # Label-specific metrics
        print("\n📈 LABEL-SPECIFIC METRICS")
        print("-" * 70)
        
        # True-fact accuracy (Verbalization ability)
        true_acc = self.compute_accuracy(
            self.results['label_true']['correct'],
            self.results['label_true']['total']
        )
        print(f"{'True-Fact Accuracy':<30} "
              f"{self.results['label_true']['correct']:<12} "
              f"{self.results['label_true']['total']:<12} "
              f"{true_acc:>6.2f}%")
        print(f"  → Measures verbalization ability")
        
        # False-fact accuracy (Hallucination resistance)
        false_acc = self.compute_accuracy(
            self.results['label_false']['correct'],
            self.results['label_false']['total']
        )
        print(f"{'False-Fact Accuracy':<30} "
              f"{self.results['label_false']['correct']:<12} "
              f"{self.results['label_false']['total']:<12} "
              f"{false_acc:>6.2f}%")
        print(f"  → Measures hallucination resistance")
        
        print("-" * 70)
        
        # Error breakdown
        print("\n⚠️  ERROR BREAKDOWN")
        print("-" * 70)
        total_errors = self.results['total'] - self.results['correct']
        
        error_counts = {
            k: len(v) for k, v in self.results['errors'].items()
        }
        
        # Sort by count descending
        sorted_errors = sorted(
            error_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for error_type, count in sorted_errors:
            if count > 0:
                pct = (count / total_errors * 100) if total_errors > 0 else 0
                print(f"{error_type.replace('_', ' ').title():<30} "
                      f"{count:<12} ({pct:>5.1f}%)")
        
        print("="*70)
    
    def save_detailed_results(self, output_path: str = None):
        """
        Save detailed results including error examples to JSON.
        
        Args:
            output_path: Path to save results. Defaults to results/slm2_evaluation_results.json
        """
        if output_path is None:
            output_path = self.output_file.parent.parent / 'results' / 'slm2_evaluation_results.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'summary': {
                'overall_accuracy': self.compute_accuracy(
                    self.results['correct'], 
                    self.results['total']
                ),
                'single_hop_accuracy': self.compute_accuracy(
                    self.results['hop_1']['correct'],
                    self.results['hop_1']['total']
                ),
                'two_hop_accuracy': self.compute_accuracy(
                    self.results['hop_2']['correct'],
                    self.results['hop_2']['total']
                ),
                'true_fact_accuracy': self.compute_accuracy(
                    self.results['label_true']['correct'],
                    self.results['label_true']['total']
                ),
                'false_fact_accuracy': self.compute_accuracy(
                    self.results['label_false']['correct'],
                    self.results['label_false']['total']
                ),
                'total_examples': self.results['total'],
                'correct_predictions': self.results['correct'],
                'incorrect_predictions': self.results['total'] - self.results['correct']
            },
            'detailed_counts': {
                'overall': self.results,
            },
            'error_examples': self.results['errors']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    def generate_error_report(self, output_path: str = None, max_examples: int = 5):
        """
        Generate a human-readable error report.
        
        Args:
            output_path: Path to save report. Defaults to results/slm2_error_report.txt
            max_examples: Maximum examples to show per error type
        """
        if output_path is None:
            output_path = self.output_file.parent.parent / 'results' / 'slm2_error_report.txt'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" SLM2 ERROR ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            for error_type, examples in self.results['errors'].items():
                if not examples:
                    continue
                
                f.write(f"\n{'='*80}\n")
                f.write(f"{error_type.replace('_', ' ').title()} ({len(examples)} errors)\n")
                f.write(f"{'='*80}\n\n")
                
                for i, ex in enumerate(examples[:max_examples]):
                    f.write(f"Example {i+1} (ID: {ex['id']}, Hop: {ex['hop_count']}, Label: {ex['label']})\n")
                    f.write(f"Input: {ex['input']}\n")
                    f.write(f"Gold:  {ex['gold']}\n")
                    f.write(f"Pred:  {ex['predicted']}\n")
                    f.write("-" * 80 + "\n\n")
                
                if len(examples) > max_examples:
                    f.write(f"... and {len(examples) - max_examples} more examples\n\n")
        
        print(f"📄 Error report saved to: {output_path}")


def main():
    """Main execution function."""
    # Path to output file
    output_file = Path(__file__).parent / 'slm2_output.jsonl'
    
    # Initialize evaluator
    evaluator = SLM2Evaluator(output_file)
    
    # Load predictions
    evaluator.load_predictions()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Print results
    evaluator.print_results()
    
    # Save detailed results
    evaluator.save_detailed_results()
    
    # Generate error report
    evaluator.generate_error_report(max_examples=10)
    
    print("\n✅ Evaluation complete!\n")


if __name__ == '__main__':
    main()
