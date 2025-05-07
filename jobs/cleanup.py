import asyncio
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, DatetimeRange
from app.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, logger

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


async def cleanup_job():
    while True:
        try:
            threshold = datetime.utcnow() - timedelta(minutes=20)

            filter_ = Filter(
                must=[
                    FieldCondition(key="timestamp", range=DatetimeRange(lt=threshold))
                ]
            )

            hits = client.scroll(
                collection_name=COLLECTION_NAME, filter=filter_, limit=100
            )

            ids_to_delete = [
                point.id for point in hits[0]
            ]  # hits[0] contiene los puntos

            if ids_to_delete:
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector={"points": ids_to_delete},
                    wait=True,
                )
                logger.info(
                    f"🧹 Cleanup ejecutado: eliminados {len(ids_to_delete)} documentos."
                )
            else:
                logger.info(
                    "🧹 Cleanup ejecutado: no se encontraron documentos para eliminar."
                )

        except Exception as e:
            logger.error("❌ Error en cleanup_job: %s", e, exc_info=True)

        await asyncio.sleep(60 * 5)  # Espera 5 minutos antes de volver a ejecutar
