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
                        range=MatchValue(value=str(threshold.isoformat())),
                    )
                ]
            )

            # Obtener los puntos que cumplen el filtro
            response = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=[],  # No se necesita un vector de consulta para este caso
                filter=filter_,
                limit=100,  # Obtener los primeros 100 documentos que coincidan con el filtro
            )
            ids_to_delete = [point.id for point in response.result]  # Extraer los ids

            if ids_to_delete:
                # Eliminar los puntos seleccionados
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=ids_to_delete,  # Aquí estamos pasando los puntos a eliminar
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

        await asyncio.sleep(60 * 5)  # Corre cada 5 minutos
