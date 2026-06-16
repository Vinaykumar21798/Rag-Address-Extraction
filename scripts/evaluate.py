import sys
import subprocess
import argparse

def run_script(script_name, extra_args):
    cmd = [sys.executable, "-m", f"scripts.{script_name}"] + extra_args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return f"Error running {script_name}: {res.stderr.strip()}"
    return res.stdout.strip()

def parse_score(output_str):
    try:
        last_line = output_str.strip().split("\n")[-1]
        if ":" in last_line:
            parts = last_line.split(":")
            return float(parts[-1].strip())
    except Exception:
        pass
    return None

def get_last_line(output_str):
    if not output_str:
        return ""
    lines = [line.strip() for line in output_str.split("\n") if line.strip()]
    return lines[-1] if lines else ""

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline scorecard by running separate evaluation scripts.")
    parser.add_argument(
        "--sample-answerable", type=int, default=None,
        help="Limit the number of answerable questions to evaluate. Default runs all."
    )
    parser.add_argument(
        "--sample-unanswerable", type=int, default=None,
        help="Limit the number of unanswerable questions to evaluate. Default runs all."
    )
    args = parser.parse_args()
    extra_ans = []
    if args.sample_answerable is not None:
        extra_ans = ["--sample-answerable", str(args.sample_answerable)]
    extra_unans = []
    if args.sample_unanswerable is not None:
        extra_unans = ["--sample-unanswerable", str(args.sample_unanswerable)]

    print("Running separate evaluation scripts...")
    recall_out = run_script("recall_at_4", extra_ans)
    recall_rerank_out = run_script("recall_reranked", extra_ans)
    recall_rewritten_out = run_script("recall_rewritten", extra_ans)
    mrr_out = run_script("mrr", extra_ans)
    mrr_rerank_out = run_script("mrr_reranked", extra_ans)
    mrr_rewritten_out = run_script("mrr_rewritten", extra_ans)
    accuracy_out = run_script("accuracy", extra_ans)
    refusal_out = run_script("refusalrate", extra_unans)

    print("\n========================= SCORECARD =========================")
    print(get_last_line(recall_out))
    print(get_last_line(mrr_out))
    print("-" * 60)
    print(get_last_line(recall_rerank_out))
    print(get_last_line(mrr_rerank_out))
    print("-" * 60)
    print(get_last_line(recall_rewritten_out))
    print(get_last_line(mrr_rewritten_out))
    print("-" * 60)
    print(get_last_line(accuracy_out))
    print(get_last_line(refusal_out))
    print("=============================================================")

    # Parse and find the weakest metric
    metrics_dict = {}
    def add_metric(name, output_str):
        score = parse_score(output_str)
        if score is not None:
            metrics_dict[name] = score

    add_metric("Base Retrieval Recall@4", recall_out)
    add_metric("Base Retrieval MRR", mrr_out)
    add_metric("Reranked Retrieval Recall@4", recall_rerank_out)
    add_metric("Reranked Retrieval MRR", mrr_rerank_out)
    add_metric("Reranked + Rewritten Recall@4", recall_rewritten_out)
    add_metric("Reranked + Rewritten MRR", mrr_rewritten_out)
    add_metric("RAG Answer Accuracy", accuracy_out)
    add_metric("RAG Refusal Rate", refusal_out)

    if metrics_dict:
        weakest = min(metrics_dict, key=metrics_dict.get)
        print(f"\nWeakest Metric: {weakest} (Score: {metrics_dict[weakest]:.4f})")
        print("=============================================================")

if __name__ == "__main__":
    main()
