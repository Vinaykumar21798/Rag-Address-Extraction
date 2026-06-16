from app.llm import generate

def rewrite_query(
    question: str
):
    prompt = f"""
Rewrite the question for document retrieval.

Keep the meaning exactly the same.
Do not introduce new facts.
Do not change postal mail into email.
Return only the rewritten query.


Question:
{question}

Return only the rewritten query.
"""
    return generate(
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=50
    )
