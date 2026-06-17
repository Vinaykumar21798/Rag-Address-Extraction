import csv
from pathlib import Path
from app.retriever import retrieve

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))

answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()][:5]
unanswerable_rows = [row for row in reader if not row.get("answer_file") or not row["answer_file"].strip()]

print("Retrieval Scores for Answerable Questions:")
for idx, row in enumerate(answerable_rows, 1):
    question = row["question"]
    res = retrieve(question, top_k=4)
    print(f"\n[{idx}/5] Q: {question}")
    for r in res:
        print(f"  Score: {r['score']:.4f} | File: {r['filename']} | Chunk snippet: {r['chunk'][:60]}")

print("\n==================================================")
print("Retrieval Scores for Unanswerable Questions:")
for idx, row in enumerate(unanswerable_rows, 1):
    question = row["question"]
    res = retrieve(question, top_k=4)
    print(f"\n[{idx}/8] Q: {question}")
    for r in res:
        print(f"  Score: {r['score']:.4f} | File: {r['filename']} | Chunk snippet: {r['chunk'][:60]}")
