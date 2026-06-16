from sentence_transformers import CrossEncoder
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(
    question: str,
    chunks: list[str]
):
    pairs = [
        (question, chunk)
        for chunk in chunks
    ]
    scores = model.predict(
        pairs
    )
    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked
