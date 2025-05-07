from qdrant_client.models import Distance, VectorParams
from config import COLLECTION_NAME, logger
from services.qdrant_client import client
from services.qa_chain import vectorstore


def insert_documents(docs):
    try:
        if COLLECTION_NAME not in [
            c.name for c in client.get_collections().collections
        ]:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        vectorstore.add_documents(docs)
    except ValueError as e:
        logger.error("❌ Error al subir archivo: %s", e, exc_info=True)
