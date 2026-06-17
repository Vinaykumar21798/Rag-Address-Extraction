import csv
import argparse
from pathlib import Path
from app.vector_store import search

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def evaluate_mrr():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-answerable", type=int, default=None)
    args, _ = parser.parse_known_args()
    if not CSV_PATH.exists():
        print("MRR: 0.0000")
        return
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if not answerable_rows:
        print("MRR: 0.0000")
        return
    mrr_sum = 0.0
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        answer_file = row["answer_file"].strip()
        print(f"[{idx}/{len(answerable_rows)}] Q: {question}")
        try:
            res = search(question, top_k=4)
            metas = res["metadatas"][0] if res.get("metadatas") else []
            files = [m.get("filename") or m.get("source") or "" for m in metas]
            if answer_file in files:
                rank = files.index(answer_file) + 1
                reciprocal = 1.0 / rank
                mrr_sum += reciprocal
                print(f"  -> MRR: Rank {rank} (Reciprocal: {reciprocal:.4f}, Files: {files})")
            else:
                print(f"  -> MRR: MISS (Reciprocal: 0.0000, Expected: '{answer_file}', Got: {files})")
        except Exception as e:
            print(f"  -> MRR: ERROR ({e})")
    score = mrr_sum / len(answerable_rows)
    print(f"Base Retrieval MRR                      : {score:.4f}")

if __name__ == "__main__":
    evaluate_mrr()
