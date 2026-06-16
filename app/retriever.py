from app.vector_store import search, collection
from app.reranker import rerank

def retrieve(
    question: str,
    top_k: int = 4
):
    count = collection.count()
    if count == 0:
        return []
    top_n = min(20, count)
    results = search(
        question,
        top_k=top_n
    )
    chunks = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    chunk_to_meta = {}
    for chunk, meta in zip(chunks, metadatas):
        chunk_to_meta[chunk] = meta or {}
    ranked = rerank(
        question,
        chunks
    )
    results_list = []
    for chunk, score in ranked[:top_k]:
        meta = chunk_to_meta.get(chunk, {})
        filename = (
            meta.get("filename")
            or meta.get("source")
            or ""
        )
        results_list.append({
            "filename": filename,
            "chunk": chunk,
            "score": float(score)
        })
    return results_list
