import chromadb
from app.embeddings import embed

class DummyEmbeddingFunction:
    def name(self) -> str:
        return "DummyEmbeddingFunction"

    def __call__(self, texts):
        return [[0.0] * 384 for _ in texts]
client = chromadb.PersistentClient(
    path="vector_db"
)
try:
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=DummyEmbeddingFunction()
    )

except ValueError:
    collection = client.get_or_create_collection(
        name="documents"
    )

def search(
    query: str,
    top_k: int = 4
):
    query_embedding = embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results
