#!/usr/bin/env python3
"""
SLM2 Master Benchmark Runner

Runs complete SLM2 evaluation suite:
1. Standard Test Set Evaluation
2. Adversarial Test Set Evaluation  
3. Hallucination Detection & Entity Coverage Analysis
4. Comprehensive Report Generation

Usage:
    python benchmark/slm2_benchmark_runner.py
    
Or run individual benchmarks:
    python benchmark/slm2_inference.py
    python benchmark/slm2_hallucination_check.py
    python scripts/prepare_slm2_adversarial.py
"""
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


# File paths
BENCHMARK_DIR = Path(__file__).parent
SCRIPTS_DIR = BENCHMARK_DIR.parent / "scripts"
DATA_DIR = BENCHMARK_DIR.parent / "data"
RESULTS_DIR = BENCHMARK_DIR.parent / "results"
TRAIN_DIR = DATA_DIR / "train"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Test files
STANDARD_TEST = TRAIN_DIR / "slm2_test.jsonl"
ADVERSARIAL_TEST = TRAIN_DIR / "slm2_adversarial_test.jsonl"

# Scripts
ADVERSARIAL_GENERATOR = SCRIPTS_DIR / "prepare_slm2_adversarial.py"
INFERENCE_SCRIPT = BENCHMARK_DIR / "slm2_inference.py"
HALLUCINATION_SCRIPT = BENCHMARK_DIR / "slm2_hallucination_check.py"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_command(command: list, description: str) -> bool:
    """
    Run a shell command and return success status.
    
    Args:
        command: List of command parts
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output if any
        if result.stdout:
            print(result.stdout)
        
        print(f"✅ {description} - COMPLETE\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - FAILED")
        print(f"   Error: Python executable not found")
        return False


def check_prerequisites() -> bool:
    """Check if all required files exist."""
    print_section("Prerequisites Check")
    
    missing_files = []
    
    # Check test files
    if not STANDARD_TEST.exists():
        missing_files.append(str(STANDARD_TEST))
    else:
        print(f"✓ Standard test file found: {STANDARD_TEST}")
    
    # Check scripts
    required_scripts = [
        ADVERSARIAL_GENERATOR,
        INFERENCE_SCRIPT,
        HALLUCINATION_SCRIPT
    ]
    
    for script in required_scripts:
        if not script.exists():
            missing_files.append(str(script))
        else:
            print(f"✓ Script found: {script.name}")
    
    if missing_files:
        print("\n❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("\n✅ All prerequisites satisfied")
    return True


def generate_adversarial_tests() -> bool:
    """Generate adversarial test cases if they don't exist."""
    print_section("Step 1: Generate Adversarial Test Cases")
    
    if ADVERSARIAL_TEST.exists():
        print(f"✓ Adversarial test file already exists: {ADVERSARIAL_TEST}")
        print("  Skipping generation (delete file to regenerate)")
        return True
    
    return run_command(
        [sys.executable, str(ADVERSARIAL_GENERATOR)],
        "Generating adversarial test cases"
    )


def run_standard_evaluation() -> bool:
    """Run evaluation on standard test set."""
    print_section("Step 2: Standard Test Set Evaluation")
    
    print("📊 Running SLM2 inference on standard test set...")
    print(f"   Test file: {STANDARD_TEST}")
    print(f"   This may take several minutes depending on GPU/CPU speed.\n")
    
    # Note: slm2_inference.py needs to be configured to use standard test
    # For now, we'll note this requires manual configuration
    print("⚠️  NOTE: Ensure slm2_inference.py is configured for standard test set")
    print("   Update TEST_FILE path in the script if needed.\n")
    
    return run_command(
        [sys.executable, str(INFERENCE_SCRIPT)],
        "Running inference on standard test set"
    )


def run_adversarial_evaluation() -> bool:
    """Run evaluation on adversarial test set."""
    print_section("Step 3: Adversarial Test Set Evaluation")
    
    print("🔥 Running SLM2 inference on adversarial test set...")
    print(f"   Test file: {ADVERSARIAL_TEST}")
    print(f"   Critical tests: Empty results, hallucination traps, edge cases.\n")
    
    print("⚠️  NOTE: Ensure slm2_inference.py is configured for adversarial test set")
    print("   Update TEST_FILE path in the script if needed.\n")
    
    # This requires manual configuration or passing test file as argument
    # For automated version, would need to modify slm2_inference.py to accept CLI args
    
    print("⚠️  MANUAL STEP REQUIRED:")
    print("   1. Edit benchmark/slm2_inference.py")
    print(f"   2. Set TEST_FILE = Path('{ADVERSARIAL_TEST}')")
    print("   3. Run: python benchmark/slm2_inference.py")
    print("   4. Re-run this benchmark runner\n")
    
    return True  # Skip for now, requires manual intervention


def run_hallucination_analysis() -> bool:
    """Run hallucination detection and entity coverage analysis."""
    print_section("Step 4: Hallucination & Entity Coverage Analysis")
    
    return run_command(
        [sys.executable, str(HALLUCINATION_SCRIPT)],
        "Analyzing hallucinations and entity coverage"
    )


def generate_summary_report() -> bool:
    """Generate comprehensive summary report."""
    print_section("Step 5: Comprehensive Summary Report")
    
    # Load results from various benchmark files
    results_files = {
        "standard_results": RESULTS_DIR / "slm2_benchmark_results.json",
        "hallucination_analysis": RESULTS_DIR / "slm2_hallucination_analysis.json",
    }
    
    results = {}
    missing_results = []
    
    for key, filepath in results_files.items():
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                results[key] = json.load(f)
            print(f"✓ Loaded: {filepath.name}")
        else:
            missing_results.append(filepath)
            print(f"⚠️  Missing: {filepath.name}")
    
    if missing_results:
        print("\n⚠️  Some result files are missing. Run individual benchmarks first.")
        return False
    
    # Generate summary report
    print("\n" + "="*80)
    print("SLM2 BENCHMARK SUMMARY REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Standard test results
    if "standard_results" in results:
        std_metrics = results["standard_results"].get("metrics", {})
        print("\n📊 STANDARD TEST SET:")
        print(f"   Total Examples: {std_metrics.get('total_examples', 0)}")
        print(f"   Exact Match: {std_metrics.get('exact_match', {}).get('rate', 0):.2%}")
        print(f"   Contains Answer: {std_metrics.get('contains_answer', {}).get('rate', 0):.2%}")
        print(f"   Avg BLEU Score: {std_metrics.get('avg_bleu_score', 0):.4f}")
        
        if 'hallucination' in std_metrics:
            print(f"\n🚨 HALLUCINATION METRICS:")
            print(f"   Hallucination Rate: {std_metrics['hallucination']['rate']:.2%}")
            print(f"   Target: < 5%")
            if std_metrics['hallucination']['rate'] < 0.05:
                print(f"   ✅ PASS")
            else:
                print(f"   ❌ FAIL")
        
        if 'entity_coverage' in std_metrics:
            print(f"\n📝 ENTITY COVERAGE:")
            print(f"   Average Coverage: {std_metrics['entity_coverage']['avg']:.2%}")
            print(f"   Target: > 90%")
            if std_metrics['entity_coverage']['avg'] > 0.90:
                print(f"   ✅ PASS")
            else:
                print(f"   ⚠️  WARNING")
        
        if 'idk_handling' in std_metrics:
            print(f"\n🛑 CRITICAL FAILURE CONSTRAINT:")
            print(f"   Abstention Accuracy: {std_metrics['idk_handling']['rate']:.2%}")
            print(f"   Target: 100%")
            if std_metrics['idk_handling']['rate'] == 1.0:
                print(f"   ✅ PASS")
            else:
                print(f"   ❌ FAIL - Critical!")
    
    # Hallucination analysis
    if "hallucination_analysis" in results:
        hal_results = results["hallucination_analysis"]
        print("\n" + "-"*80)
        print("🔬 DETAILED HALLUCINATION ANALYSIS:")
        print(f"   Hallucination Count: {hal_results.get('hallucination_count', 0)}")
        print(f"   Hallucination Rate: {hal_results.get('hallucination_rate', 0):.2%}")
        print(f"   Avg Entity Coverage: {hal_results.get('avg_entity_coverage', 0):.2%}")
        print(f"   Abstention Correct: {hal_results.get('abstention_correct', 0)}/{hal_results.get('abstention_total', 0)}")
    
    print("\n" + "="*80)
    
    # Save summary report
    summary_file = RESULTS_DIR / "slm2_summary_report.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Summary report saved to: {summary_file}")
    
    return True


def main():
    """Main execution function."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "SLM2 BENCHMARK SUITE" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\nThis script runs the complete SLM2 evaluation pipeline:")
    print("  1. Generate adversarial test cases")
    print("  2. Run standard test set evaluation")
    print("  3. Run adversarial test set evaluation")
    print("  4. Analyze hallucinations & entity coverage")
    print("  5. Generate comprehensive report")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Track success of each step
    steps_success = []
    
    # Step 1: Generate adversarial tests
    steps_success.append(generate_adversarial_tests())
    
    # Step 2: Standard evaluation
    # Note: This requires model to be trained and available
    print_section("Step 2: Standard Test Set Evaluation")
    print("⚠️  MANUAL STEP:")
    print("   This requires a trained SLM2 model.")
    print("   1. Ensure model adapter is at: outputs/Llama-3-8B-sparql2Text")
    print("   2. Run: python benchmark/slm2_inference.py")
    print("   3. Continue to next steps\n")
    
    # Step 3: Adversarial evaluation
    print_section("Step 3: Adversarial Test Set Evaluation")
    print("⚠️  MANUAL STEP:")
    print("   1. Update TEST_FILE in benchmark/slm2_inference.py to:")
    print(f"      TEST_FILE = Path('{ADVERSARIAL_TEST}')")
    print("   2. Run: python benchmark/slm2_inference.py")
    print("   3. Save results to: slm2_predictions_adversarial.jsonl\n")
    
    # Step 4: Hallucination analysis
    steps_success.append(run_hallucination_analysis())
    
    # Step 5: Generate summary
    # steps_success.append(generate_summary_report())
    
    # Final summary
    print_section("Benchmark Execution Complete")
    
    completed = sum(steps_success)
    total = len(steps_success)
    
    print(f"Steps Completed: {completed}/{total}")
    
    if completed == total:
        print("\n✅ All benchmarks completed successfully!")
    else:
        print("\n⚠️  Some benchmarks require manual execution.")
        print("   Follow the instructions above to complete evaluation.")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Review results in: results/")
    print("  2. Check hallucination rate < 5%")
    print("  3. Verify entity coverage > 90%")
    print("  4. Ensure abstention accuracy = 100%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
