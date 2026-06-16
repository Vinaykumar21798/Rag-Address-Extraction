import sys
try:
    from app.exceptions import LLMUnavailable
    from app.llm import generate

except Exception as e:
    print(f"LLMUnavailable: {e}")
    sys.exit(1)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python llm.py "your prompt"')
        sys.exit(1)
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
        sys.exit(1)

    except Exception as exc:
        print(f"LLMUnavailable: {exc}")
        sys.exit(1)

