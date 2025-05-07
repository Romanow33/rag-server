import os
from dotenv import load_dotenv
import logging

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TEMP_DIR = "/tmp"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.makedirs(TEMP_DIR, exist_ok=True)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


async def startup_check():
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        _ = client.get_collections()
        logger.info("✅ Conexión a Qdrant exitosa.")
    except Exception as e:
        logger.error("❌ Error al conectar a Qdrant: %s", e, exc_info=True)

    if not os.path.exists(TEMP_DIR) or not os.access(TEMP_DIR, os.W_OK):
        raise RuntimeError(f"❌ No se puede escribir en {TEMP_DIR}.")
