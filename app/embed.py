# from langchain.embeddings.base import Embeddings
# from sentence_transformers import SentenceTransformer

# class CustomHFEmbedding(Embeddings):
#     def __init__(self, model_name: str = "BAAI/bge-m3"):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         return [self.model.encode(text).tolist() for text in texts]

#     def embed_query(self, text: str) -> list[float]:
#         return self.model.encode(text).tolist()


from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
    chunk_size=50 # embed 50 chunk 1 lần 
)