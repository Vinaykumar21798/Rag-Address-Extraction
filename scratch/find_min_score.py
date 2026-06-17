import csv
from pathlib import Path
from app.retriever import retrieve

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))

answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]

min_score = 9999.0
min_q = ""
for idx, row in enumerate(answerable_rows, 1):
    q = row["question"]
    res = retrieve(q, top_k=4)
    if res:
        max_score = max(r["score"] for r in res)
        print(f"Q: {q[:50]}... | Max Score: {max_score:.4f}")
        if max_score < min_score:
            min_score = max_score
            min_q = q
    else:
        print(f"Q: {q[:50]}... | NO RESULTS")

print(f"\nMinimum retrieval score among answerable questions: {min_score:.4f} for query: {min_q}")
