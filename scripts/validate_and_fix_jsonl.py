#!/usr/bin/env python3
"""
Validate and fix JSONL files by identifying and reporting malformed lines.
"""
import json
import sys
import argparse
from pathlib import Path


def validate_jsonl(filepath, fix=False):
    """
    Validate a JSONL file and optionally create a fixed version.
    
    Args:
        filepath: Path to the JSONL file
        fix: If True, create a fixed version with valid lines only
    """
    filepath = Path(filepath)
    errors = []
    valid_lines = []
    
    print(f"Validating: {filepath}")
    print("-" * 80)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                data = json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError as e:
                errors.append({
                    'line': i,
                    'error': str(e),
                    'content': line[:200] + ('...' if len(line) > 200 else '')
                })
                print(f"❌ Error on line {i}:")
                print(f"   {e}")
                print(f"   Content preview: {line[:200]}")
                print()
    
    print("-" * 80)
    if errors:
        print(f"❌ Found {len(errors)} error(s) in {len(valid_lines) + len(errors)} lines")
        print(f"✅ Valid lines: {len(valid_lines)}")
        
        if fix:
            fixed_path = filepath.parent / f"{filepath.stem}_fixed{filepath.suffix}"
            with open(fixed_path, 'w', encoding='utf-8') as f:
                for line in valid_lines:
                    f.write(line + '\n')
            print(f"\n✅ Created fixed file: {fixed_path}")
            print(f"   Contains {len(valid_lines)} valid lines")
    else:
        print(f"✅ All {len(valid_lines)} lines are valid JSON!")
    
    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate JSONL files')
    parser.add_argument('file', help='Path to JSONL file')
    parser.add_argument('--fix', action='store_true', 
                       help='Create a fixed version with valid lines only')
    
    args = parser.parse_args()
    
    is_valid = validate_jsonl(args.file, fix=args.fix)
    sys.exit(0 if is_valid else 1)
