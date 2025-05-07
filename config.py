import os
from dotenv import load_dotenv
import logging
from qdrant_client.models import Distance, VectorParams
from services.qdrant_client import client

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

    try:
        collections = client.get_collections().collections
        logger.info("✅ Conexión a Qdrant exitosa.")

        # Verifica si existe la colección, si no, la crea
        if COLLECTION_NAME not in [c.name for c in collections]:
            logger.info("📁 La colección '%s' no existe. Creándola…", COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("✅ Colección '%s' creada exitosamente.", COLLECTION_NAME)
        else:
            logger.info("📁 La colección '%s' ya existe.", COLLECTION_NAME)

    except Exception as e:
        logger.error("❌ Error al conectar o inicializar Qdrant: %s", e, exc_info=True)
        raise RuntimeError("No se pudo conectar a Qdrant.")
