from app.chunker import chunk_text
from app.vector_store import collection
from app.embeddings import embed

def ingest_document(
    filename: str,
    text: str
):
    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks):
        chunk_id = (
            f"{filename}#{idx}"
        )
        collection.upsert(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[
                embed(chunk)
            ],
            metadatas=[
                {
                    "filename": filename
                }
            ]
        )
    return len(chunks)

