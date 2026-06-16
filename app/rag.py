import time
import json
import re
from pydantic import BaseModel
from app.retriever import retrieve
from app.llm import generate
from app.database.db import create_rag_log
from app.exceptions import LLMUnavailable

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    context_found: bool

def answer_question(question: str, db=None) -> AskResponse:
    start_time = time.time()
    results = retrieve(question, top_k=4)
    
    # Hallucination guardrail: if the best retrieved chunk relevance score is extremely low, decline immediately
    if not results or max(res["score"] for res in results) < -1.0:
        return AskResponse(answer="I don't know", sources=[], context_found=False)
        
    context_parts = []
    retrieved_filenames = []
    for res in results:
        fn = res["filename"]
        chunk_text = res["chunk"]
        context_parts.append(f"Source file: {fn}\nContent: {chunk_text}")
        if fn not in retrieved_filenames:
            retrieved_filenames.append(fn)
    context = "\n\n".join(context_parts)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict, factual Q&A assistant. Answer the question using ONLY the provided context.\n"
                "If the context does not explicitly, directly and clearly contain the answer to the question, you MUST reply exactly 'I don't know'.\n"
                "Do not extrapolate, assume, or make up details. Do not use outside knowledge."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Format your response as a JSON object with keys:\n"
                "\"answer\" (string, or exactly \"I don't know\"),\n"
                "\"sources\" (list of strings representing filenames used, or empty list if answer is \"I don't know\"),\n"
                "\"context_found\" (boolean, true if answer is found in context, false otherwise).\n\n"
                "JSON response:"
            )
        }
    ]
    response_text = ""
    try:
        response_text = generate(messages, max_tokens=60)

    except LLMUnavailable:
        raise
    answer = "I don't know"
    sources = []
    context_found = False
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            validated = AskResponse.model_validate(data)
            answer = validated.answer
            sources = validated.sources
            context_found = validated.context_found

        else:
            if "i don't know" in response_text.lower() or "don't know" in response_text.lower():
                answer = "I don't know"
                sources = []
                context_found = False

            else:
                answer = response_text.strip()
                sources = [fn for fn in retrieved_filenames if fn.lower() in answer.lower()]
                if not sources:
                    sources = retrieved_filenames
                context_found = True

    except Exception:
        if "i don't know" in response_text.lower() or "don't know" in response_text.lower():
            answer = "I don't know"
            sources = []
            context_found = False

        else:
            answer = response_text.strip()
            sources = retrieved_filenames
            context_found = True
    if "i don't know" in answer.lower():
        answer = "I don't know"
        sources = []
        context_found = False
    latency = time.time() - start_time
    if db is not None:
        try:
            retrieved_files_str = json.dumps(retrieved_filenames)
            create_rag_log(db, question, retrieved_files_str, latency)

        except Exception:
            pass
    return AskResponse(answer=answer, sources=sources, context_found=context_found)
