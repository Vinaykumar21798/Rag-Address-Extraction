import os
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from app.database.db import init_db, get_db, SessionLocal
from app.database.models import Document, Address
from app.vector_store import collection
from app.ingestion import ingest_document
from app.rag import answer_question
DB_FILE = Path("registry.db")
VECTOR_DB_DIR = Path("vector_db")
CORPUS_DIR = Path("corpus")

def reset_databases():
    print("Resetting SQL database...")
    if DB_FILE.exists():
        try:
            DB_FILE.unlink()
            print("  Removed existing registry.db")

        except Exception as e:
            print(f"  Warning: Could not delete registry.db: {e}")
    init_db()
    print("  SQL database initialized successfully.")
    print("Resetting Chroma vector store...")
    try:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"  Deleted {len(existing_ids)} existing chunks from Chroma store.")

        else:
            print("  Chroma store was already empty.")

    except Exception as e:
        print(f"  Warning during vector store reset: {e}")

def ingest_corpus():
    print("\nIngesting and indexing corpus files...")
    if not CORPUS_DIR.exists():
        print(f"  Error: {CORPUS_DIR} directory not found.")
        return
    db = SessionLocal()
    try:
        total_chunks = 0
        files = list(CORPUS_DIR.glob("*"))
        print(f"  Found {len(files)} files to ingest.")
        for file_path in files:
            if file_path.is_file():
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                doc = Document(
                    filename=file_path.name,
                    size_bytes=file_path.stat().st_size,
                    status="processed",
                    content=text
                )
                db.add(doc)
                db.commit()
                chunks_count = ingest_document(file_path.name, text)
                total_chunks += chunks_count
                print(f"    - Ingested: {file_path.name} ({chunks_count} chunks)")
        print(f"  Ingestion complete. Total chunks: {total_chunks}")

    finally:
        db.close()

def print_stats():
    print("\n" + "=" * 15 + " SYSTEM STATISTICS " + "=" * 15)
    db = SessionLocal()
    try:
        docs_count = db.query(Document).count()
        addr_count = db.query(Address).count()
        chunks_count = collection.count()
        print(f"  Extracted Documents in SQL  : {docs_count}")
        print(f"  Normalized Addresses in SQL : {addr_count}")
        print(f"  Document Chunks in VectorDB : {chunks_count}")

    finally:
        db.close()
    print("=" * 50)

def run_demo_questions():
    questions = [
        "What forwarding address does the Office of Records notice give for future mail?",
        "What is the total amount due on invoice 4471?",
        "Who is responsible for writing integration tests after the Q3 planning meeting?"
    ]
    print("\nRunning RAG QA demonstration questions...")
    print("=" * 60)
    db = SessionLocal()
    try:
        for idx, q in enumerate(questions, 1):
            print(f"\nQuestion {idx}: {q}")
            result = answer_question(q, db=db)
            print(f"Answer    : {result.answer}")
            print(f"Sources   : {', '.join(result.sources) if result.sources else 'None'}")
            print(f"Context Found: {result.context_found}")
            print("-" * 60)

    finally:
        db.close()
if __name__ == "__main__":
    reset_databases()
    ingest_corpus()
    print_stats()
    run_demo_questions()

