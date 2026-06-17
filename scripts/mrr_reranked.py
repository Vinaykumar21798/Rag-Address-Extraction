import csv
import argparse
from pathlib import Path
from app.retriever import retrieve

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def evaluate_mrr_reranked():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-answerable", type=int, default=None)
    args, _ = parser.parse_known_args()
    if not CSV_PATH.exists():
        print("Reranked MRR: 0.0000")
        return
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if not answerable_rows:
        print("Reranked MRR: 0.0000")
        return
    mrr_sum = 0.0
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        answer_file = row["answer_file"].strip()
        print(f"[{idx}/{len(answerable_rows)}] Q: {question}")
        try:
            res = retrieve(question, top_k=4)
            files = [r["filename"] for r in res]
            if answer_file in files:
                rank = files.index(answer_file) + 1
                reciprocal = 1.0 / rank
                mrr_sum += reciprocal
                print(f"  -> RERANKED MRR: Rank {rank} (Reciprocal: {reciprocal:.4f}, Files: {files})")
            else:
                print(f"  -> RERANKED MRR: MISS (Reciprocal: 0.0000, Expected: '{answer_file}', Got: {files})")
        except Exception as e:
            print(f"  -> RERANKED MRR: ERROR ({e})")
    score = mrr_sum / len(answerable_rows)
    print(f"Reranked Retrieval MRR                  : {score:.4f}")

if __name__ == "__main__":
    evaluate_mrr_reranked()
