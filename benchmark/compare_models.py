"""
Model Comparison Script

Compares evaluation results from multiple models side-by-side.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_results(results_file: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_models(model_results: Dict[str, str]):
    """
    Compare multiple models' results.
    
    Args:
        model_results: Dict mapping model_name -> results_file_path
    """
    # Load all results
    all_results = {}
    for model_name, results_file in model_results.items():
        results_path = Path(results_file)
        if not results_path.exists():
            print(f"⚠️  Warning: Results file not found for {model_name}: {results_file}")
            continue
        all_results[model_name] = load_results(results_file)
    
    if not all_results:
        print("❌ No results files found!")
        return
    
    # Print comparison table
    print("\n" + "="*100)
    print(" MODEL COMPARISON - SLM2 BENCHMARK")
    print("="*100)
    
    # Main metrics
    print("\n📊 ACCURACY METRICS")
    print("-" * 100)
    
    header = f"{'Metric':<30}"
    for model_name in all_results.keys():
        header += f"{model_name:>20}"
    print(header)
    print("-" * 100)
    
    metrics = [
        ('Overall Accuracy', 'overall_accuracy'),
        ('Single-Hop Accuracy', 'single_hop_accuracy'),
        ('2-Hop Accuracy', 'two_hop_accuracy'),
        ('True-Fact Accuracy', 'true_fact_accuracy'),
        ('False-Fact Accuracy', 'false_fact_accuracy'),
    ]
    
    for metric_label, metric_key in metrics:
        row = f"{metric_label:<30}"
        values = []
        for model_name, results in all_results.items():
            value = results['summary'][metric_key]
            values.append(value)
            row += f"{value:>19.2f}%"
        print(row)
        
        # Highlight best performer
        best_value = max(values)
        best_models = [name for name, results in all_results.items() 
                      if results['summary'][metric_key] == best_value]
        if len(best_models) == 1:
            print(f"{'  → Best: ' + best_models[0]:<30}")
    
    print("-" * 100)
    
    # Error analysis
    print("\n⚠️  ERROR ANALYSIS")
    print("-" * 100)
    
    header = f"{'Metric':<30}"
    for model_name in all_results.keys():
        header += f"{model_name:>20}"
    print(header)
    print("-" * 100)
    
    # Calculate error counts
    for model_name, results in all_results.items():
        total = results['summary']['total_examples']
        incorrect = results['summary']['incorrect_predictions']
        
        # Count errors by type
        error_counts = {}
        for error_type, examples in results['error_examples'].items():
            error_counts[error_type] = len(examples)
    
    # Print error type distribution
    error_types = ['hallucination', 'over_verbalization', 'under_verbalization', 
                   'wrong_decision', 'false_confidence', 'multi_hop_failure', 'other']
    
    for error_type in error_types:
        row = f"{error_type.replace('_', ' ').title():<30}"
        for model_name, results in all_results.items():
            count = len(results['error_examples'].get(error_type, []))
            total_errors = results['summary']['incorrect_predictions']
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            row += f"{count:>12} ({pct:>5.1f}%)"
        print(row)
    
    print("-" * 100)
    
    # Summary statistics
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 100)
    
    header = f"{'Metric':<30}"
    for model_name in all_results.keys():
        header += f"{model_name:>20}"
    print(header)
    print("-" * 100)
    
    summary_metrics = [
        ('Total Examples', 'total_examples'),
        ('Correct Predictions', 'correct_predictions'),
        ('Incorrect Predictions', 'incorrect_predictions'),
    ]
    
    for metric_label, metric_key in summary_metrics:
        row = f"{metric_label:<30}"
        for model_name, results in all_results.items():
            value = results['summary'][metric_key]
            row += f"{value:>20}"
        print(row)
    
    print("="*100)
    
    # Determine overall winner
    print("\n🏆 OVERALL RANKING (by Overall Accuracy)")
    print("-" * 100)
    
    rankings = sorted(
        all_results.items(),
        key=lambda x: x[1]['summary']['overall_accuracy'],
        reverse=True
    )
    
    for rank, (model_name, results) in enumerate(rankings, 1):
        acc = results['summary']['overall_accuracy']
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"{medal} {model_name:<25} {acc:>6.2f}%")
    
    print("="*100)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare multiple model evaluation results")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='../results',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Model names to compare (e.g., slm2 gpt-oss)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.models:
        # User specified models
        model_results = {}
        for model_name in args.models:
            results_file = results_dir / f"{model_name}_evaluation_results.json"
            model_results[model_name] = str(results_file)
    else:
        # Auto-detect all evaluation results
        model_results = {}
        for results_file in results_dir.glob("*_evaluation_results.json"):
            model_name = results_file.stem.replace('_evaluation_results', '')
            model_results[model_name] = str(results_file)
    
    if not model_results:
        print("❌ No evaluation results found!")
        print(f"\nLooked in: {results_dir}")
        print("\nRun evaluations first:")
        print("  python slm2_evaluation.py")
        print("  python gpt_oss_evaluate.py --output-file gpt_oss_output.jsonl --model-name gpt-oss")
        return
    
    print(f"\nFound {len(model_results)} model(s) to compare:")
    for model_name in model_results.keys():
        print(f"  • {model_name}")
    
    compare_models(model_results)


if __name__ == '__main__':
    main()
