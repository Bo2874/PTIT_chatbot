from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = HuggingFaceCrossEncoder(model_name=model_name)

    def rerank(self, query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
        if not docs:
            return []

        # Tạo batch (query, doc_text)
        pairs = [(query, doc.page_content) for doc in docs]

        # Tính điểm similarity (batch predict)
        scores = self.reranker.score(pairs)

        # Gắn score vào metadata
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = float(score)

        # Sắp xếp theo score giảm dần
        docs_sorted = sorted(docs, key=lambda d: d.metadata["score"], reverse=True)

        return docs_sorted[:top_k]