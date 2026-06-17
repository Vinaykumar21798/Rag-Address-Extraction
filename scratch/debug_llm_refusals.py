import sys
from app.rag import answer_question, retrieve
from app.llm import generate

questions = [
    "Who is the chief executive officer of Riverside Office Supplies?",
    "What is the API key for the production embedding service?"
]

for q in questions:
    print(f"\n====================\nQ: {q}")
    results = retrieve(q, top_k=4)
    context_parts = []
    for res in results:
        context_parts.append(f"Source file: {res['filename']}\nContent: {res['chunk']}")
    context = "\n\n".join(context_parts)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict, factual Q&A assistant. Answer the question using ONLY the provided context.\n"
                "If the context does not contain the answer, reply exactly 'I don't know'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {q}\n\n"
                "Format your response as a JSON object matching this schema:\n"
                "{\n"
                "  \"answer\": \"your answer, or exactly 'I don't know'\",\n"
                "  \"sources\": [\"source_filename.txt\"],\n"
                "  \"context_found\": true or false\n"
                "}\n\n"
                "JSON response:"
            )
        }
    ]
    raw_response = generate(messages, max_tokens=150)
    print("--- RAW LLM RESPONSE ---")
    print(raw_response)
    print("------------------------")
    res = answer_question(q)
    print(f"Parsed Answer: {res.answer}")
    print(f"Sources: {res.sources}")
    print(f"Context Found: {res.context_found}")
