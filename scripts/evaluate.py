import csv
import argparse
from pathlib import Path
from app.retriever import retrieve
from app.rag import answer_question

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline scorecard in a single process.")
    parser.add_argument(
        "--sample-answerable", type=int, default=None,
        help="Limit the number of answerable questions to evaluate."
    )
    parser.add_argument(
        "--sample-unanswerable", type=int, default=None,
        help="Limit the number of unanswerable questions to evaluate."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Split rows into answerable and unanswerable
    answerable_rows = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    unanswerable_rows = [row for row in reader if not row.get("answer_file") or not row["answer_file"].strip()]

    # Apply limits if provided
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if args.sample_unanswerable is not None:
        unanswerable_rows = unanswerable_rows[:args.sample_unanswerable]

    print(f"Starting evaluation... (Answerable: {len(answerable_rows)}, Unanswerable: {len(unanswerable_rows)})", flush=True)

    recall_hits = 0
    mrr_sum = 0.0
    accuracy_hits = 0
    diagnoses = []

    # 1. Evaluate answerable questions
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        answer_file = row["answer_file"].strip()
        expected_keyphrase = row["expected_keyphrase"].strip()

        print(f"[{idx}/{len(answerable_rows)}] Evaluating Answerable Q: '{question[:50]}...'", flush=True)

        # Retrieve (Recall & MRR)
        try:
            retrieved = retrieve(question, top_k=4)
            files = [r["filename"] for r in retrieved]
        except Exception as e:
            print(f"  -> Retrieval error: {e}", flush=True)
            files = []

        is_recall_hit = answer_file in files
        if is_recall_hit:
            recall_hits += 1
            rank = files.index(answer_file) + 1
            mrr_sum += 1.0 / rank
        
        # Ask (Answer Accuracy)
        try:
            res = answer_question(question)
            answer_text = res.answer
        except Exception as e:
            print(f"  -> RAG Answer error: {e}", flush=True)
            answer_text = "ERROR"

        is_accuracy_hit = expected_keyphrase.lower() in answer_text.lower()
        if is_accuracy_hit:
            accuracy_hits += 1
        else:
            # Diagnosis: retrieval hit but answer miss
            if is_recall_hit:
                diagnoses.append(
                    f"  - '{question[:50]}...': right file retrieved but answer missed '{expected_keyphrase}' (Got: '{answer_text.strip()}')"
                )

    # 2. Evaluate unanswerable questions
    refusal_hits = 0
    for idx, row in enumerate(unanswerable_rows, 1):
        question = row["question"]
        print(f"[{idx}/{len(unanswerable_rows)}] Evaluating Unanswerable Q: '{question[:50]}...'", flush=True)

        try:
            res = answer_question(question)
            answer_text = res.answer
        except Exception as e:
            print(f"  -> RAG Answer error: {e}", flush=True)
            answer_text = "ERROR"

        if "i don't know" in answer_text.lower():
            refusal_hits += 1

    # Compute final metrics
    recall_score = recall_hits / len(answerable_rows) if answerable_rows else 0.0
    mrr_score = mrr_sum / len(answerable_rows) if answerable_rows else 0.0
    accuracy_score = accuracy_hits / len(answerable_rows) if answerable_rows else 0.0
    refusal_score = refusal_hits / len(unanswerable_rows) if unanswerable_rows else 0.0

    # Find weakest metric
    metrics = {
        "recall@4": recall_score,
        "MRR": mrr_score,
        "answer_accuracy": accuracy_score,
        "refusal_rate": refusal_score
    }
    weakest = min(metrics, key=metrics.get)

    # Print Scorecard
    print("\n========================================")
    print("RAG SCORECARD")
    print("========================================")
    print(f"  recall@4           {recall_score:.3f}")
    print(f"  MRR                {mrr_score:.3f}")
    print(f"  answer_accuracy    {accuracy_score:.3f}")
    print(f"  refusal_rate       {refusal_score:.3f}")
    print()
    print(f"  answerable: {len(answerable_rows)}   unanswerable: {len(unanswerable_rows)}")
    print(f"  weakest metric: {weakest} ({metrics[weakest]:.3f})")
    print()
    if diagnoses:
        print("Retrieval-hit / answer-miss diagnoses:")
        for diag in diagnoses:
            print(diag)
    else:
        print("No Retrieval-hit / answer-miss diagnoses recorded.")
    print("========================================")

if __name__ == "__main__":
    main()
