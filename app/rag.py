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
    if not results or max(res["score"] for res in results) < -6.0:
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
                "Answer exactly and directly, extracting the specific name, address, value, or fact from the text.\n"
                "Do not paraphrase or change names or numbers.\n"
                "If the context does not contain the answer, reply exactly 'I don't know'.\n"
                "Do not guess, assume, or extrapolate. If the context does not explicitly mention the specific requested information, "
                "you must reply 'I don't know'."
            )
        },
        {
            "role": "user",
            "content": (
                "Context:\n"
                "Source file: letter_dc.txt\n"
                "Content: The prior mailing address on file (742 Evergreen Terrace, Suite 4) is retired. Forwarding address: 1600 Pennsylvania Ave NW, Washington, DC 20500.\n\n"
                "Question: What forwarding address does the Office of Records notice give for future mail?\n\n"
                "Format your response as a JSON object matching this schema:\n"
                "{\n"
                "  \"answer\": \"your answer, or exactly 'I don't know'\",\n"
                "  \"sources\": [\"source_filename.txt\"],\n"
                "  \"context_found\": true or false\n"
                "}"
            )
        },
        {
            "role": "assistant",
            "content": (
                "{\n"
                "  \"answer\": \"1600 Pennsylvania Ave NW\",\n"
                "  \"sources\": [\"letter_dc.txt\"],\n"
                "  \"context_found\": true\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                "Context:\n"
                "Source file: letter_riverside.txt\n"
                "Content: Written inquiries about invoices, refunds, or address corrections should all be directed to the Accounts Team.\n"
                "Sincerely,\nThe Accounts Team\n\n"
                "Question: Who is the chief executive officer of Riverside Office Supplies?\n\n"
                "Format your response as a JSON object matching this schema:\n"
                "{\n"
                "  \"answer\": \"your answer, or exactly 'I don't know'\",\n"
                "  \"sources\": [\"source_filename.txt\"],\n"
                "  \"context_found\": true or false\n"
                "}"
            )
        },
        {
            "role": "assistant",
            "content": (
                "{\n"
                "  \"answer\": \"I don't know\",\n"
                "  \"sources\": [],\n"
                "  \"context_found\": false\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                "Context:\n"
                "Source file: incident_postmortem.md\n"
                "Content: The TLS certificate on the internal embedding endpoint expired and auto-renewal had silently failed.\n\n"
                "Question: What is the API key for the production embedding service?\n\n"
                "Format your response as a JSON object matching this schema:\n"
                "{\n"
                "  \"answer\": \"your answer, or exactly 'I don't know'\",\n"
                "  \"sources\": [\"source_filename.txt\"],\n"
                "  \"context_found\": true or false\n"
                "}"
            )
        },
        {
            "role": "assistant",
            "content": (
                "{\n"
                "  \"answer\": \"I don't know\",\n"
                "  \"sources\": [],\n"
                "  \"context_found\": false\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Format your response as a JSON object matching this schema:\n"
                "{\n"
                "  \"answer\": \"your answer, or exactly 'I don't know'\",\n"
                "  \"sources\": [\"source_filename.txt\"],\n"
                "  \"context_found\": true or false\n"
                "}"
            )
        }
    ]
    response_text = ""
    try:
        response_text = generate(messages, max_tokens=150)


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
