from pathlib import Path
import csv
from app.extractor import extract_addresses
from app.regex_extractor import extract_addresses_regex
SAMPLE_DIR = Path("sample_pdfs")
OUTPUT_CSV = "extractor_comparison.csv"
import fitz

def read_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        try:
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text

        except Exception:
            return ""
    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

def run_comparison():
    rows = []
    files = list(SAMPLE_DIR.glob("*"))
    for file_path in files:
        try:
            text = read_file(file_path)
            regex_result = extract_addresses_regex(
                text
            )
            llm_result = extract_addresses(
                text
            )
            regex_found = (
                len(
                    regex_result.get(
                        "addresses",
                        []
                    )
                ) > 0
            )
            llm_found = (
                len(
                    llm_result["addresses"][
                        "addresses"
                    ]
                ) > 0
            )
            rows.append(
                {
                    "file": file_path.name,
                    "regex_found": regex_found,
                    "llm_status":
                        llm_result["status"],
                    "llm_found": llm_found,
                }
            )
            print(
                f"{file_path.name} "
                f"| Regex={regex_found} "
                f"| LLM={llm_found}"
            )

        except Exception as e:
            rows.append(
                {
                    "file": file_path.name,
                    "regex_found": "ERROR",
                    "llm_status": "ERROR",
                    "llm_found": str(e),
                }
            )
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "regex_found",
                "llm_status",
                "llm_found",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"\nSaved report to "
        f"{OUTPUT_CSV}"
    )
if __name__ == "__main__":
    run_comparison()
