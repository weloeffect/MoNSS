# import json
# import os

# input_file = './data/train/slm2_train_data.jsonl'
# output_file = './data/train/slm2_train_data_norm.jsonl'

# def normalize_all_entities(infile, outfile):
#     print(f"Scanning {infile} for mixed types...")
    
#     with open(infile, 'r', encoding='utf-8') as f_in, \
#          open(outfile, 'w', encoding='utf-8') as f_out:
        
#         row_count = 0
#         fixed_fields_count = 0
        
#         for line in f_in:
#             try:
#                 data = json.loads(line)
                
#                 # Check if 'entities' dictionary exists
#                 if 'entities' in data and isinstance(data['entities'], dict):
                    
#                     # Iterate over every key in the 'entities' dictionary
#                     # (e.g., 'star', 'writer', 'director', 'producer', 'author', etc.)
#                     for key, value in data['entities'].items():
                        
#                         # If the value is a simple string, wrap it in a list
#                         if isinstance(value, str):
#                             data['entities'][key] = [value]
#                             fixed_fields_count += 1
                            
#                 # Write the corrected object back to the new file
#                 json.dump(data, f_out)
#                 f_out.write('\n')
#                 row_count += 1
                
#             except json.JSONDecodeError:
#                 print(f"Skipping invalid JSON at line {row_count + 1}")
#                 continue

#     print(f"Processed {row_count} rows.")
#     print(f"Fixed {fixed_fields_count} fields that were strings instead of lists.")
#     print(f"Saved final data to: {outfile}")

# if __name__ == "__main__":
#     if os.path.exists(input_file):
#         normalize_all_entities(input_file, output_file)
#     else:
#         print(f"Error: Could not find {input_file}. Please check the filename.")


#!/usr/bin/env python3
# """
# Fix malformed JSONL file by identifying and removing bad lines.
# """
# import json
# from pathlib import Path

# INPUT_FILE = Path(__file__).parent / "data" / "train" / "slm1_adversarial_test.jsonl"
# OUTPUT_FILE = Path(__file__).parent / "data" / "train" / "slm1_adversarial_test_fixed.jsonl"


# def fix_jsonl():
#     print("=" * 80)
#     print("Fixing adversarial JSONL file")
#     print("=" * 80)
#     print("Input: " + str(INPUT_FILE))
#     print("Output: " + str(OUTPUT_FILE))
    
#     valid_lines = []
#     invalid_lines = []
    
#     with open(INPUT_FILE, "r", encoding="utf-8") as f:
#         for line_num, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue
            
#             try:
#                 data = json.loads(line)
#                 # Validate required fields
#                 if "input" in data and "output" in data:
#                     valid_lines.append(data)
#                 else:
#                     print("Line " + str(line_num) + ": Missing required fields")
#                     invalid_lines.append((line_num, line, "Missing fields"))
#             except json.JSONDecodeError as e:
#                 print("Line " + str(line_num) + ": JSON error - " + str(e))
#                 print("  Content: " + line[:100] + "...")
#                 invalid_lines.append((line_num, line, str(e)))
    
#     print("")
#     print("=" * 80)
#     print("Results:")
#     print("  Valid lines: " + str(len(valid_lines)))
#     print("  Invalid lines: " + str(len(invalid_lines)))
    
#     if len(valid_lines) > 0:
#         with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#             for data in valid_lines:
#                 f.write(json.dumps(data, ensure_ascii=False) + "\n")
#         print("")
#         print("Fixed file saved to: " + str(OUTPUT_FILE))
#         print("")
#         print("To use the fixed file, run:")
#         print("  mv slm1_adversarial_test.jsonl slm1_adversarial_test_backup.jsonl")
#         print("  mv slm1_adversarial_test_fixed.jsonl slm1_adversarial_test.jsonl")
    
#     if len(invalid_lines) > 0:
#         print("")
#         print("Invalid lines details:")
#         for line_num, content, error in invalid_lines[:5]:
#             print("  Line " + str(line_num) + ": " + error)
#             print("    " + content[:80] + "...")
    
#     print("=" * 80)


# if __name__ == "__main__":
#     fix_jsonl()