from huggingface_hub import login, upload_folder

# Login to Hugging Face (will prompt for token)
login()

# Upload the trained LoRA adapters
upload_folder(
    folder_path="outputs/Llama-3-8B-text2sparql-qlora",
    repo_id="weloSai/Llama3-8b-text2Sparql",
    repo_type="model"
)

print("✅ Model successfully pushed to: https://huggingface.co/weloSai/Llama3-8b-text2Sparql")