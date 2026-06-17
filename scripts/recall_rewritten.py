import csv
import argparse
from pathlib import Path
from app.query_rewriter import rewrite_query
from app.retriever import retrieve

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def evaluate_recall_rewritten():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-answerable", type=int, default=None)
    args, _ = parser.parse_known_args()
    if not CSV_PATH.exists():
        print("Reranked + Rewritten Recall@4: 0.0000")
        return
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if not answerable_rows:
        print("Reranked + Rewritten Recall@4: 0.0000")
        return
    hits = 0
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        answer_file = row["answer_file"].strip()
        print(f"[{idx}/{len(answerable_rows)}] Q: {question}")
        try:
            rewritten = rewrite_query(question)
            res = retrieve(rewritten, top_k=4)
            files = [r["filename"] for r in res]
            is_hit = answer_file in files
            if is_hit:
                hits += 1
                print(f"  -> REWRITTEN RECALL@4: HIT (Files: {files})")
            else:
                print(f"  -> REWRITTEN RECALL@4: MISS (Expected: '{answer_file}', Got: {files})")
        except Exception as e:
            print(f"  -> REWRITTEN RECALL@4: ERROR ({e})")
    score = hits / len(answerable_rows)
    print(f"Reranked + Rewritten Recall@4           : {score:.4f}")

if __name__ == "__main__":
    evaluate_recall_rewritten()
