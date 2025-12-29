"""
Fix SLM1 Model Predictions
--------------------------
Post-processing script to fix spacing issues in model-generated SPARQL queries.

Issues fixed:
1. SELECT?var -> SELECT ?var (missing space after SELECT)
2. dbo:pred?var -> dbo:pred ?var (missing space before variable after predicate)
3. ?var. -> ?var . (missing space before period)
"""

import json
import re
from pathlib import Path


def fix_sparql_spacing(sparql: str) -> str:
    """
    Fix common spacing issues in generated SPARQL queries.
    
    Args:
        sparql: Raw SPARQL query with potential spacing issues
        
    Returns:
        Fixed SPARQL query with proper spacing
    """
    if not sparql:
        return sparql
    
    fixed = sparql
    
    # Fix 1: SELECT?var -> SELECT ?var
    fixed = re.sub(r'\bSELECT\?', 'SELECT ?', fixed)
    
    # Fix 2: ASK{ -> ASK {
    fixed = re.sub(r'\bASK\{', 'ASK {', fixed)
    
    # Fix 3: WHERE{ -> WHERE {
    fixed = re.sub(r'\bWHERE\{', 'WHERE {', fixed)
    
    # Fix 4: predicate?var -> predicate ?var (for dbo: and dbr: predicates)
    # Handle cases like dbo:director?person -> dbo:director ?person
    fixed = re.sub(r'(dbo:\w+)\?(\w+)', r'\1 ?\2', fixed)
    fixed = re.sub(r'(dbr:\w+)\?(\w+)', r'\1 ?\2', fixed)
    
    # Fix 5: ?var. -> ?var .
    fixed = re.sub(r'\?(\w+)\.(\s|$|\})', r'?\1 .\2', fixed)
    
    # Fix 6: Multiple spaces -> single space
    fixed = re.sub(r'  +', ' ', fixed)
    
    # Fix 7: Space before closing brace if missing after period
    fixed = re.sub(r'\.\}', '. }', fixed)
    
    # Fix 8: Ensure space after opening brace
    fixed = re.sub(r'\{\s*(\w)', r'{ \1', fixed)
    
    # Fix 9: Convert prefixed URIs with special characters to full URI form
    # This handles: parentheses, apostrophes, and other special chars
    # dbr:Name_(something) -> <http://dbpedia.org/resource/Name_(something)>
    def expand_problematic_uri(match):
        prefix = match.group(1)
        local = match.group(2)
        if prefix == 'dbr':
            return f'<http://dbpedia.org/resource/{local}>'
        elif prefix == 'dbo':
            return f'<http://dbpedia.org/ontology/{local}>'
        return match.group(0)
    
    # Match prefixed names containing parentheses (with possible trailing content)
    fixed = re.sub(r'\b(dbr|dbo):([A-Za-z0-9_]+\([^)]+\)[A-Za-z0-9_]*)', expand_problematic_uri, fixed)
    
    # Match prefixed names containing apostrophes
    fixed = re.sub(r"\b(dbr|dbo):([A-Za-z0-9_]*'[A-Za-z0-9_']+)", expand_problematic_uri, fixed)
    
    # Fix 10: Handle space before parentheses in entity names
    # dbr:Name (something) -> <http://dbpedia.org/resource/Name_(something)>
    def fix_space_before_parens(match):
        prefix = match.group(1)
        name = match.group(2)
        parens = match.group(3)
        local = f"{name}_({parens})"  # Convert space to underscore
        if prefix == 'dbr':
            return f'<http://dbpedia.org/resource/{local}>'
        elif prefix == 'dbo':
            return f'<http://dbpedia.org/ontology/{local}>'
        return match.group(0)
    
    fixed = re.sub(r'\b(dbr|dbo):([A-Za-z0-9_]+) \(([^)]+)\)', fix_space_before_parens, fixed)
    
    return fixed.strip()


def validate_sparql_syntax(sparql: str) -> tuple:
    """
    Check if SPARQL query has valid syntax.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        from rdflib.plugins.sparql import prepareQuery
        prepareQuery(sparql)
        return True, None
    except Exception as e:
        return False, str(e)


def fix_predictions_file(input_path: str, output_path: str = None) -> dict:
    """
    Fix all predictions in a JSON file.
    
    Args:
        input_path: Path to predictions JSON file
        output_path: Path for fixed predictions (defaults to input_path with _fixed suffix)
        
    Returns:
        Statistics dictionary
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed.json"
    else:
        output_path = Path(output_path)
    
    # Load predictions
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle nested structure: {"total_predictions": N, "predictions": [...]}
    if isinstance(data, dict) and 'predictions' in data:
        predictions = data['predictions']
        is_nested = True
    else:
        predictions = data
        is_nested = False
    
    stats = {
        'total': len(predictions),
        'fixed': 0,
        'valid_before': 0,
        'valid_after': 0,
        'still_invalid': [],
        'changes': []
    }
    
    # Fix each prediction
    for i, pred in enumerate(predictions):
        if 'predicted_sparql' not in pred:
            continue
            
        original = pred['predicted_sparql']
        fixed = fix_sparql_spacing(original)
        
        # Track changes
        if original != fixed:
            stats['fixed'] += 1
            stats['changes'].append({
                'index': i,
                'original': original,
                'fixed': fixed
            })
        
        # Validate before
        valid_before, _ = validate_sparql_syntax(original)
        if valid_before:
            stats['valid_before'] += 1
        
        # Update prediction
        pred['predicted_sparql'] = fixed
        pred['original_prediction'] = original  # Keep original for reference
        
        # Validate after
        valid_after, error = validate_sparql_syntax(fixed)
        if valid_after:
            stats['valid_after'] += 1
        else:
            stats['still_invalid'].append({
                'index': i,
                'query': fixed,
                'error': error
            })
    
    # Reconstruct data structure
    if is_nested:
        data['predictions'] = predictions
        output_data = data
    else:
        output_data = predictions
    
    # Save fixed predictions
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    return stats, output_path


def main():
    """Main function to fix all prediction files."""
    print("=" * 60)
    print("Fixing SLM1 Predictions")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    
    # Files to fix
    prediction_files = [
        base_dir / "data" / "train" / "slm1_predictions_standard.json",
        base_dir / "data" / "train" / "slm1_predictions_adversarial.json"
    ]
    
    for pred_file in prediction_files:
        if not pred_file.exists():
            print(f"\n⚠ File not found: {pred_file.name}")
            continue
            
        print(f"\n{'=' * 50}")
        print(f"Processing: {pred_file.name}")
        print("=" * 50)
        
        stats, output_path = fix_predictions_file(pred_file)
        
        print(f"\nTotal predictions: {stats['total']}")
        print(f"Predictions modified: {stats['fixed']}")
        print(f"\nValidity:")
        print(f"  Before fix: {stats['valid_before']}/{stats['total']} "
              f"({100*stats['valid_before']/max(1,stats['total']):.1f}%)")
        print(f"  After fix:  {stats['valid_after']}/{stats['total']} "
              f"({100*stats['valid_after']/max(1,stats['total']):.1f}%)")
        
        if stats['changes']:
            print(f"\nSample changes (first 3):")
            for change in stats['changes'][:3]:
                print(f"\n  [{change['index']}] Original: {change['original']}")
                print(f"      Fixed:    {change['fixed']}")
        
        if stats['still_invalid']:
            print(f"\n⚠ Still invalid after fix: {len(stats['still_invalid'])}")
            for inv in stats['still_invalid'][:3]:
                print(f"  [{inv['index']}] {inv['query'][:60]}...")
                print(f"      Error: {inv['error'][:80]}")
        
        print(f"\n✓ Fixed predictions saved to: {output_path.name}")
    
    print("\n" + "=" * 60)
    print("Done! Run benchmark scripts on the *_fixed.json files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
