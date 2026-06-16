import csv
import time
import torch
import argparse
from pathlib import Path
torch.set_num_threads(2)
from app.vector_store import search, collection
from app.retriever import retrieve
from app.query_rewriter import rewrite_query
from app.rag import answer_question
CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline scorecard.")
    parser.add_argument(
        "--sample-answerable", type=int, default=None,
        help="Limit the number of answerable questions to evaluate. Default runs all."
    )
    parser.add_argument(
        "--sample-unanswerable", type=int, default=None,
        help="Limit the number of unanswerable questions to evaluate. Default runs all."
    )
    args = parser.parse_args()
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist.")
        return
    count = collection.count()
    if count == 0:
        print("Warning: The Chroma database collection is empty! Run demo.py or reindex first.")
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    total_answerable = [row for row in reader if row.get("answer_file") and row["answer_file"].strip()]
    total_unanswerable = [row for row in reader if not row.get("answer_file") or not row["answer_file"].strip()]
    answerable_rows = total_answerable
    unanswerable_rows = total_unanswerable
    if args.sample_answerable is not None:
        answerable_rows = answerable_rows[:args.sample_answerable]
    if args.sample_unanswerable is not None:
        unanswerable_rows = unanswerable_rows[:args.sample_unanswerable]
    print(f"Loaded {len(reader)} questions from {CSV_PATH}")
    print(f"  Answerable questions to evaluate: {len(answerable_rows)} (Total available: {len(total_answerable)})")
    print(f"  Unanswerable questions to evaluate: {len(unanswerable_rows)} (Total available: {len(total_unanswerable)})")
    print("=" * 60)
    base_hits = 0
    base_mrr_sum = 0.0
    reranked_hits = 0
    reranked_mrr_sum = 0.0
    rewritten_hits = 0
    rewritten_mrr_sum = 0.0
    accuracy_hits = 0
    diagnoses = []
    print("\n--- Evaluating Answerable Questions (Retrieval & Accuracy) ---")
    for idx, row in enumerate(answerable_rows, 1):
        question = row["question"]
        answer_file = row["answer_file"].strip()
        expected_keyphrase = row["expected_keyphrase"].strip()
        print(f"[{idx}/{len(answerable_rows)}] Q: {question}")
        try:
            base_res = search(question, top_k=4)
            base_metas = base_res["metadatas"][0] if base_res.get("metadatas") else []
            base_files = [m.get("filename") or m.get("source") or "" for m in base_metas]

        except Exception:
            base_files = []
        if answer_file in base_files:
            base_hits += 1
            rank = base_files.index(answer_file) + 1
            base_mrr_sum += 1.0 / rank
        try:
            reranked_res = retrieve(question, top_k=4)
            reranked_files = [r["filename"] for r in reranked_res]

        except Exception:
            reranked_files = []
        if answer_file in reranked_files:
            reranked_hits += 1
            rank = reranked_files.index(answer_file) + 1
            reranked_mrr_sum += 1.0 / rank
        try:
            rewritten_q = rewrite_query(question)
            rewritten_res = retrieve(rewritten_q, top_k=4)
            rewritten_files = [r["filename"] for r in rewritten_res]

        except Exception:
            rewritten_files = []
        if answer_file in rewritten_files:
            rewritten_hits += 1
            rank = rewritten_files.index(answer_file) + 1
            rewritten_mrr_sum += 1.0 / rank
        try:
            rag_res = answer_question(question)
            answer = rag_res.answer
            sources = rag_res.sources

        except Exception as e:
            answer = f"ERROR: {e}"
            sources = []
        is_correct = expected_keyphrase.lower() in answer.lower()
        if is_correct:
            accuracy_hits += 1
            print(f"  -> ACCURACY: HIT (Sources: {sources})")

        else:
            print(f"  -> ACCURACY: MISS (Expected: '{expected_keyphrase}', Got: '{answer}')")
            if answer_file in reranked_files:
                diagnoses.append(
                    f"DIAGNOSIS for '{question}': Retrieval succeeded (found in {answer_file}), but model answer was wrong (Expected: '{expected_keyphrase}', Got: '{answer}')."
                )
    print("\n--- Evaluating Unanswerable Questions (Refusal Rate) ---")
    correct_refusals = 0
    for idx, row in enumerate(unanswerable_rows, 1):
        question = row["question"]
        print(f"[{idx}/{len(unanswerable_rows)}] Q: {question}")
        try:
            rag_res = answer_question(question)
            answer = rag_res.answer

        except Exception as e:
            answer = f"ERROR: {e}"
        if "i don't know" in answer.lower():
            correct_refusals += 1
            print("  -> REFUSAL: PASS")

        else:
            print(f"  -> REFUSAL: FAIL (Answered: '{answer}')")
    num_ans = len(answerable_rows)
    num_unans = len(unanswerable_rows)
    base_recall = base_hits / num_ans if num_ans > 0 else 0.0
    base_mrr = base_mrr_sum / num_ans if num_ans > 0 else 0.0
    reranked_recall = reranked_hits / num_ans if num_ans > 0 else 0.0
    reranked_mrr = reranked_mrr_sum / num_ans if num_ans > 0 else 0.0
    rewritten_recall = rewritten_hits / num_ans if num_ans > 0 else 0.0
    rewritten_mrr = rewritten_mrr_sum / num_ans if num_ans > 0 else 0.0
    accuracy = accuracy_hits / num_ans if num_ans > 0 else 0.0
    refusal_rate = correct_refusals / num_unans if num_unans > 0 else 0.0
    print("\n" + "=" * 25 + " SCORECARD " + "=" * 25)
    print(f"Base Retrieval Recall@4                 : {base_recall:.4f}")
    print(f"Base Retrieval MRR                      : {base_mrr:.4f}")
    print("-" * 60)
    print(f"Reranked Retrieval Recall@4             : {reranked_recall:.4f}")
    print(f"Reranked Retrieval MRR                  : {reranked_mrr:.4f}")
    print("-" * 60)
    print(f"Reranked + Rewritten Recall@4           : {rewritten_recall:.4f}")
    print(f"Reranked + Rewritten MRR                : {rewritten_mrr:.4f}")
    print("-" * 60)
    print(f"RAG Answer Accuracy                     : {accuracy:.4f}")
    print(f"RAG Refusal Rate                        : {refusal_rate:.4f}")
    print("=" * 60)
    if diagnoses:
        print("\n--- Diagnostic Details (Retrieval Hit, Answer Miss) ---")
        for diag in diagnoses:
            print(diag)
    metrics_dict = {
        "Reranked Recall@4": reranked_recall,
        "Reranked MRR": reranked_mrr,
        "Answer Accuracy": accuracy,
        "Refusal Rate": refusal_rate
    }
    weakest = min(metrics_dict, key=metrics_dict.get)
    print(f"\nWeakest Metric: {weakest} (Score: {metrics_dict[weakest]:.4f})")
    print("=" * 60)
if __name__ == "__main__":
    run_evaluation()
