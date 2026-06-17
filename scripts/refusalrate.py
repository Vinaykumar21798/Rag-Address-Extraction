import csv
from pathlib import Path

from app.rag import answer_question

CSV_PATH = Path("evaluation/rag_questions_sample_key.csv")


def evaluate_refusal_rate():
    if not CSV_PATH.exists():
        print("RAG Refusal Rate : 0.0000")
        return

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Week 3: unanswerable questions have blank answer_file
    unanswerable_rows = [
        row
        for row in rows
        if not row.get("answer_file")
        or not row["answer_file"].strip()
    ]

    if not unanswerable_rows:
        print("RAG Refusal Rate : 0.0000")
        return

    refusals = 0

    for idx, row in enumerate(unanswerable_rows, start=1):

        question = row["question"]

        print(
            f"[{idx}/{len(unanswerable_rows)}] "
            f"Q: {question}"
        )

        try:
            response = answer_question(question)

            answer = response.answer.strip().lower()

            # Week 3 requirement
            if answer == "i don't know":
                refusals += 1
                print("  -> REFUSAL: PASS")
            else:
                print(
                    f"  -> REFUSAL: FAIL "
                    f"(Answered: '{response.answer}')"
                )

        except Exception as exc:
            print(
                f"  -> REFUSAL: ERROR ({exc})"
            )

    refusal_rate = refusals / len(unanswerable_rows)

    print()
    print(
        f"RAG Refusal Rate : "
        f"{refusal_rate:.4f}"
    )


if __name__ == "__main__":
    evaluate_refusal_rate()