import csv
import argparse
from pathlib import Path
from app.vector_store import search

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def evaluate_recall():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-answerable", type=int, default=None)
    args, _ = parser.parse_known_args()
    if not CSV_PATH.exists():
        print("Recall@4: 0.0000")
        return
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if not answerable_rows:
        print("Recall@4: 0.0000")
        return
    hits = 0
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        answer_file = row["answer_file"].strip()
        print(f"[{idx}/{len(answerable_rows)}] Q: {question}")
        try:
            res = search(question, top_k=4)
            metas = res["metadatas"][0] if res.get("metadatas") else []
            files = [m.get("filename") or m.get("source") or "" for m in metas]
            is_hit = answer_file in files
            if is_hit:
                hits += 1
                print(f"  -> RECALL@4: HIT (Files: {files})")
            else:
                print(f"  -> RECALL@4: MISS (Expected: '{answer_file}', Got: {files})")
        except Exception as e:
            print(f"  -> RECALL@4: ERROR ({e})")
    score = hits / len(answerable_rows)
    print(f"Base Retrieval Recall@4                 : {score:.4f}")

if __name__ == "__main__":
    evaluate_recall()
