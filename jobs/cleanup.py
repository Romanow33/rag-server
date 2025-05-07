import asyncio
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, logger

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


async def cleanup_job():
    while True:
        try:
            threshold = datetime.utcnow() - timedelta(minutes=20)
            filter_ = Filter(
                must=[
                    FieldCondition(
                        key="timestamp",
                        match=MatchValue(value=str(threshold.isoformat())),
                    )
                ]
            )

            client.delete(collection_name=COLLECTION_NAME, filter=filter_, wait=True)

            logger.info("🧹 Cleanup ejecutado: documentos viejos eliminados.")
        except Exception as e:
            logger.error("❌ Error en cleanup_job: %s", e, exc_info=True)

        await asyncio.sleep(60 * 5)
