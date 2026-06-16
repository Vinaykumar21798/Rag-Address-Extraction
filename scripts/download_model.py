import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "local_model", "qwen2.5-0.5b-instruct")
)

def download_model():
    print(f"Downloading model '{MODEL_NAME}' from Hugging Face...")
    print("This might take a few minutes depending on your internet connection.")
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="cpu"
        )
        print(f"Saving tokenizer and model to local directory: {LOCAL_DIR}")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_DIR)
        model.save_pretrained(LOCAL_DIR)
        print("Download and local save completed successfully!")

    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    download_model()

