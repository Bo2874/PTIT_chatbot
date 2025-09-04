from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", use_gpu=True):
        device = "cuda" if use_gpu else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: list[Document], top_k: int = 5, batch_size: int = 8) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]

        # Batch prediction trên GPU nếu device="cuda"
        scores = self.model.predict(pairs, batch_size=batch_size)

        # Gắn score vào metadata
        for i, doc in enumerate(docs):
            doc.metadata["score"] = float(scores[i])

        reranked = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
        return reranked[:top_k]
    

