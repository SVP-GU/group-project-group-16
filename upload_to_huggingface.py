from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def upload_model_to_hub(
    local_model_path,
    repo_name="swedish-news-classifier",
    organization="Mirac1999",
    private=False
):
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create the full repository name
    full_repo_name = f"{organization}/{repo_name}"
    
    try:
        # Create the repository
        create_repo(
            repo_id=full_repo_name,
            private=private,
            exist_ok=True
        )
        
        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        
        # Push the model and tokenizer to the hub
        model.push_to_hub(full_repo_name)
        tokenizer.push_to_hub(full_repo_name)
        
        # If there's an eval_results.json file, upload it too
        eval_results_path = os.path.join(local_model_path, "eval_results.json")
        if os.path.exists(eval_results_path):
            api.upload_file(
                path_or_fileobj=eval_results_path,
                path_in_repo="eval_results.json",
                repo_id=full_repo_name
            )
        
        print(f"Successfully uploaded model to: https://huggingface.co/{full_repo_name}")
        return True
        
    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        return False

if __name__ == "__main__":
    # Path to your latest model
    model_path = "grupp_sista (kopia)/trained_models/bert_model_20250525_001044"
    
    # Upload the model
    upload_model_to_hub(
        local_model_path=model_path,
        repo_name="swedish-news-classifier",
        organization="Mirac1999",
        private=False  # Set to True if you want a private repository
    ) 