from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 1. Point to the Base Model used in training
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"
# 2. Point to the folder where main_slm2.py saved the adapter
ADAPTER_PATH = "outputs/Qwen2.5-1.5B-SPARQL2TEXT"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and attach the trained LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()

# Example SPARQL query and result for testing
# ...existing code...
sparql_input = "SPARQL Query:\nSELECT ?director WHERE {\n  dbr:Day_for_Night_(film) dbo:director ?director .\n}\n\nSPARQL Result:\n- director: François Truffaut"

# Use the exact template format. 
# Assuming standard Alpaca format based on your headers.
prompt = f"""### Instruction:
Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.
If the result is empty, say "I don't know."
Do not add information.

### Input:
{sparql_input}

### Output:
"""

print(prompt)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)


with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,      # IMPORTANT: deterministic output
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id
    )

print("running inference on result interpreter")
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Inference is over")
print(output)


# Extract the natural# filepath: c:\Users\walee\Documents\CPS2\Research_NeSy\MoNSSpecialists\inf_slm2.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 1. Point to the Base Model used in training
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
# 2. Point to the folder where main_slm2.py saved the adapter
ADAPTER_PATH = "outputs/Llama-3-8B-sparql2Text"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and attach the trained LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()

# Example SPARQL query and result for testing
# ...existing code...
sparql_input = "SPARQL Query:\nSELECT ?director WHERE {\n  dbr:Day_for_Night_(film) dbo:director ?director .\n}\n\nSPARQL Result:\n- director: François Truffaut"

# Use the exact template format. 
# Assuming standard Alpaca format based on your headers.
prompt = f"""### Instruction:
Write a natural-language answer to the SPARQL query using ONLY the SPARQL results.
If the result is empty, say "I don't know."
Do not add information.

### Input:
{sparql_input}

### Output:
"""

print(prompt)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)


with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,      # IMPORTANT: deterministic output
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id
    )

print("running inference on result interpreter")
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Inference is over")
print(output)


# Extract the natural