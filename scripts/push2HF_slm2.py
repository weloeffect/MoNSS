from huggingface_hub import login, upload_folder

# Login to Hugging Face (will prompt for token)
login()

# Upload the trained LoRA adapters
upload_folder(
    folder_path="outputs/Qwen2.5-1.5B-SPARQL2TEXT",
    repo_id="weloSai/Qwen2.5-1.5B-SPARQL2TEXT",
    repo_type="model"
)

print("✅ Model successfully pushed to: https://huggingface.co/weloSai/Llama3-8b-SPARQL2Text")