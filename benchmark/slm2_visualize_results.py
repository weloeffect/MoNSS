"""
SLM2 Results Visualization
Creates simple text-based visualizations of evaluation metrics
"""

def print_bar_chart(label, value, max_value=100, width=50):
    """Print a simple text-based bar chart."""
    filled = int((value / max_value) * width)
    bar = '█' * filled + '░' * (width - filled)
    print(f"{label:30} {bar} {value:6.2f}%")


def main():
    print("\n" + "="*80)
    print(" SLM2 BENCHMARK - VISUAL SUMMARY")
    print("="*80)
    
    # Overall metrics
    print("\n📊 ACCURACY COMPARISON")
    print("-" * 80)
    print_bar_chart("Overall Accuracy", 77.64)
    print_bar_chart("Single-Hop (1-hop)", 77.00)
    print_bar_chart("2-Hop Accuracy", 78.14)
    print()
    print_bar_chart("True-Fact Accuracy", 75.80)
    print_bar_chart("False-Fact Accuracy", 79.67)
    print("-" * 80)
    
    # Error breakdown
    print("\n⚠️  ERROR TYPE DISTRIBUTION (of 789 total errors)")
    print("-" * 80)
    print_bar_chart("False Confidence", 43.2, max_value=100, width=40)
    print_bar_chart("Other", 38.4, max_value=100, width=40)
    print_bar_chart("Multi-Hop Failure", 8.1, max_value=100, width=40)
    print_bar_chart("Wrong Decision", 6.0, max_value=100, width=40)
    print_bar_chart("Under-Verbalization", 4.3, max_value=100, width=40)
    print("-" * 80)
    
    # Hop count distribution
    print("\n📈 DATASET COMPOSITION")
    print("-" * 80)
    print(f"1-Hop Examples:    1,561 (44.3%) {'█' * 22}{'░' * 28}")
    print(f"2-Hop Examples:    1,967 (55.7%) {'█' * 28}{'░' * 22}")
    print()
    print(f"True Facts:        1,851 (52.5%) {'█' * 26}{'░' * 24}")
    print(f"False Facts:       1,677 (47.5%) {'█' * 24}{'░' * 26}")
    print("-" * 80)
    
    # Success/Failure breakdown
    print("\n✅ SUCCESS vs ❌ FAILURE")
    print("-" * 80)
    print(f"Correct:           2,739 (77.6%) {'█' * 39}{'░' * 11}")
    print(f"Incorrect:           789 (22.4%) {'█' * 11}{'░' * 39}")
    print("-" * 80)
    
    # Insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 80)
    print("✅ STRENGTHS:")
    print("  • Better at detecting unknowns (79.67%) than verbalizing facts (75.80%)")
    print("  • 2-hop reasoning (78.14%) performs better than 1-hop (77.00%)")
    print("  • Low multi-hop failure rate (only 8.1% of errors)")
    print()
    print("⚠️  WEAKNESSES:")
    print("  • False confidence is the main error (43.2% of all errors)")
    print("  • Semantic mismatches in verbalization (38.4% categorized as 'Other')")
    print("  • Some under-verbalization with placeholders (4.3%)")
    print("-" * 80)
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
