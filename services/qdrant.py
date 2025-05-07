from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, logger
from services.embeddings import embedding
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def insert_documents(docs):
    try:
        if COLLECTION_NAME not in [
            c.name for c in client.get_collections().collections
        ]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        vectorstore = LangchainQdrant(
            client=client, collection_name=COLLECTION_NAME, embeddings=embedding
        )
        vectorstore.add_documents(docs)
    except ValueError as e:
        logger.error("❌ Error al subir archivo: %s", e, exc_info=True)
