import json
import re
from pathlib import Path

def extract_entities_from_sparql_and_output(sparql_input, output_text):
    """
    Extract entities from SPARQL query results and natural language output.
    Returns a dictionary of entity types and their values.
    """
    entities = {}
    
    # Extract the subject/object from SPARQL query (entity being queried)
    # Pattern: dbr:<EntityName> dbo:<property>
    subject_match = re.search(r'dbr:([^\s]+)\s+dbo:', sparql_input)
    if subject_match:
        subject_entity = subject_match.group(1).replace('_', ' ')
        # Clean up parenthetical disambiguations
        subject_entity = re.sub(r'_\(.*?\)', lambda m: ' ' + m.group(0).replace('_', ' '), subject_entity)
        subject_entity = subject_entity.strip()
    
    # Extract property being queried
    property_match = re.search(r'SELECT \?(\w+)', sparql_input)
    property_name = property_match.group(1) if property_match else None
    
    # Extract results from SPARQL Result section
    result_pattern = r'-\s+(\w+):\s+(.+?)(?=\n-|\n\n|$)'
    results = re.findall(result_pattern, sparql_input, re.MULTILINE)
    
    # Build entities dictionary
    if results:
        property_values = []
        for prop, value in results:
            # Clean up the value
            cleaned_value = value.strip()
            property_values.append(cleaned_value)
        
        # Determine the subject entity type based on patterns
        if subject_match:
            # Determine entity type from context
            if '(film)' in subject_entity or 'film' in subject_entity.lower():
                entities['film'] = subject_entity
            elif '(book)' in subject_entity or 'autobiography' in subject_entity.lower():
                entities['book'] = subject_entity
            else:
                # Default to generic subject
                entities['subject'] = subject_entity
        
        # Add the property values
        if property_name:
            if len(property_values) == 1:
                entities[property_name] = property_values[0]
            else:
                entities[property_name] = property_values
    
    return entities


def process_training_file(input_file, output_file):
    """
    Process the training JSONL file and add entities field to each entry.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    modified_entries = []
    
    for line in lines:
        entry = json.loads(line.strip())
        
        # Extract entities from input and output
        entities = extract_entities_from_sparql_and_output(
            entry['input'], 
            entry['output']
        )
        
        # Add entities field to entry
        entry['entities'] = entities
        
        modified_entries.append(entry)
    
    # Write modified entries to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in modified_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(modified_entries)} entries")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    input_file = Path("c:/Users/walee/Documents/CPS2/Research_NeSy/MoNSSpecialists/data/train/slm2_train.jsonl")
    output_file = Path("c:/Users/walee/Documents/CPS2/Research_NeSy/MoNSSpecialists/data/train/slm2_train_with_entities.jsonl")
    
    process_training_file(input_file, output_file)
    
    # Show a sample
    print("\nSample entry:")
    with open(output_file, 'r', encoding='utf-8') as f:
        first_entry = json.loads(f.readline())
        print(json.dumps(first_entry, indent=2, ensure_ascii=False))
