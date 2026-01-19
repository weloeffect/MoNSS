from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "weloSai/Llama3-8b-text2Sparql"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()


prompt = """
Translate the following into a SPARQL query:

Where was Albert Einstein born and in which country?

"""



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

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)


sparql = output.split("### SPARQL:")[-1].strip()
print("Generated SPARQL:\n", sparql)