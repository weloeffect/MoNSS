"""
Complete Benchmarking Pipeline

Runs the complete workflow:
1. Test setup
2. Run inference
3. Evaluate results
4. Compare with other models
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    # Use the same Python interpreter that's running this script
    python_exe = sys.executable
    cmd = cmd.replace("python ", f'"{python_exe}" ')
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def main():
    """Run complete benchmarking pipeline."""
    
    print("="*70)
    print(" COMPLETE BENCHMARKING PIPELINE")
    print("="*70)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n❌ OPENROUTER_API_KEY not set!")
        print("\nSet it with:")
        print("  PowerShell: $env:OPENROUTER_API_KEY = 'your-key-here'")
        print("  CMD:        set OPENROUTER_API_KEY=your-key-here")
        print("  Bash:       export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)
    
    print(f"\n✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Configuration
    MODEL_NAME = "gpt-oss"
    MODEL_ID = "openai/gpt-oss-120b:free"
    OUTPUT_FILE = "gpt_oss_output.jsonl"
    USE_REASONING = True  # Set to False for standard mode
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Reasoning: {'Enabled' if USE_REASONING else 'Disabled'}")
    
    input("\nPress Enter to start or Ctrl+C to cancel...")
    
    # Step 1: Test setup
    if not run_command(
        "python test_gpt_oss_setup.py",
        "Testing setup (5 examples)"
    ):
        print("\n⚠️  Setup test failed, but continuing anyway...")
    
    # Step 2: Run inference
    reasoning_flag = "--reasoning" if USE_REASONING else ""
    if not run_command(
        f"python gpt_oss_inference.py {reasoning_flag} --output-file {OUTPUT_FILE} --model {MODEL_ID}",
        "Running inference on full dataset (this will take 15-30 minutes)"
    ):
        print("\n❌ Inference failed! Aborting.")
        sys.exit(1)
    
    # Step 3: Evaluate
    if not run_command(
        f"python gpt_oss_evaluate.py --output-file {OUTPUT_FILE} --model-name {MODEL_NAME}",
        "Evaluating predictions"
    ):
        print("\n❌ Evaluation failed! Aborting.")
        sys.exit(1)
    
    # Step 4: Compare models
    if Path("../results/slm2_evaluation_results.json").exists():
        if not run_command(
            f"python compare_models.py --models slm2 {MODEL_NAME}",
            "Comparing with SLM2"
        ):
            print("\n⚠️  Comparison failed, but results are still saved.")
    else:
        print("\n⚠️  SLM2 results not found, skipping comparison.")
    
    # Summary
    print("\n" + "="*70)
    print(" 🎉 PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  • benchmark/{OUTPUT_FILE}")
    print(f"  • results/{MODEL_NAME}_evaluation_results.json")
    print(f"  • results/{MODEL_NAME}_error_report.txt")
    print("\nNext steps:")
    print("  1. Check results in results/ directory")
    print("  2. Run comparison: python compare_models.py")
    print("  3. Analyze errors in error report")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Aborted by user")
        sys.exit(1)
