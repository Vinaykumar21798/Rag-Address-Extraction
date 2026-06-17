from transformers import pipeline
from app.exceptions import LLMUnavailable
from transformers import AutoTokenizer, pipeline
import os
import torch
torch.set_num_threads(4)
LOCAL_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "local_model", "qwen2.5-0.5b-instruct")
)
if os.path.isdir(LOCAL_MODEL_PATH):
    MODEL_NAME = LOCAL_MODEL_PATH

else:
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = None
_generator = None
if os.environ.get("MOCK_LLM") == "true":
    pass

else:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        _generator = pipeline(
            task="text-generation",
            model=MODEL_NAME,
            trust_remote_code=True,
            model_kwargs={"dtype": "auto"}
        )

    except Exception as e:
        raise LLMUnavailable(
            f"Unable to load model '{MODEL_NAME}': {e}"
        )

def generate(messages, max_tokens=128):
    if os.environ.get("MOCK_LLM") == "true":
        prompt_content = messages[-1]["content"] if messages else ""
        if "extract every postal address" in prompt_content.lower():
            return '{"addresses": [{"street": "1600 Pennsylvania Ave NW", "city": "Washington", "state": "DC", "zip": "20500"}]}'
        return '{"answer": "1600 Pennsylvania Ave NW, Washington, DC 20500", "sources": ["letter_dc.txt"], "context_found": true}'
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        result = _generator(
            prompt,
            max_new_tokens=max_tokens,
            max_length=None,
            do_sample=False,
            return_full_text=False
        )
        return result[0]["generated_text"].strip()

    except Exception as e:
        raise LLMUnavailable(
            f"Generation failed: {e}"
        )
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(
            'Usage: python -m app.llm "your prompt"'
        )
        raise SystemExit(1)
    prompt = sys.argv[1]
    try:
        response = generate(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=20,
        )
        print(response)

    except LLMUnavailable as exc:
        print(f"LLMUnavailable: {exc}")
