from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
