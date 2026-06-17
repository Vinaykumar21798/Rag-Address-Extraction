import csv
import argparse
from pathlib import Path
from app.rag import answer_question

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def evaluate_accuracy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-answerable", type=int, default=None)
    args, _ = parser.parse_known_args()
    if not CSV_PATH.exists():
        print("Accuracy: 0.0000")
        return
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if not answerable_rows:
        print("Accuracy: 0.0000")
        return
    hits = 0
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        expected_keyphrase = row["expected_keyphrase"].strip()
        print(f"[{idx}/{len(answerable_rows)}] Q: {question}")
        try:
            res = answer_question(question)
            is_correct = expected_keyphrase.lower() in res.answer.lower()
            if is_correct:
                hits += 1
                print(f"  -> ACCURACY: HIT (Sources: {res.sources})")
            else:
                print(f"  -> ACCURACY: MISS (Expected: '{expected_keyphrase}', Got: '{res.answer}')")
        except Exception as e:
            print(f"  -> ACCURACY: ERROR ({e})")
    score = hits / len(answerable_rows)
    print(f"RAG Answer Accuracy                     : {score:.4f}")

if __name__ == "__main__":
    evaluate_accuracy()
