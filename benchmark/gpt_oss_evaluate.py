"""
GPT-OSS Evaluation Script

Evaluates GPT-OSS predictions using the same evaluation logic as SLM2.
Accepts command-line arguments for flexible evaluation of different output files.
"""

import argparse
from pathlib import Path
from slm2_evaluation import SLM2Evaluator


def main():
    """Main execution function with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-OSS or any model predictions on SLM2 test dataset"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to predictions file (e.g., gpt_oss_output.jsonl)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='../results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='gpt-oss',
        help='Model name for result files (e.g., gpt-oss, slm2)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    output_file = Path(args.output_file)
    if not output_file.exists():
        print(f"❌ Error: Output file not found: {output_file}")
        return
    
    # Initialize evaluator
    print(f"Evaluating predictions from: {output_file}")
    evaluator = SLM2Evaluator(str(output_file))
    
    # Load predictions
    evaluator.load_predictions()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Print results
    evaluator.print_results()
    
    # Save detailed results
    results_dir = Path(args.results_dir)
    results_file = results_dir / f"{args.model_name}_evaluation_results.json"
    evaluator.save_detailed_results(str(results_file))
    
    # Generate error report
    error_report_file = results_dir / f"{args.model_name}_error_report.txt"
    evaluator.generate_error_report(str(error_report_file), max_examples=10)
    
    print(f"\n✅ Evaluation complete for {args.model_name}!\n")


if __name__ == '__main__':
    main()
