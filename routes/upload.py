from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from services.documents import process_pdf
from services.qdrant import insert_documents
from sse.manager import manager

router = APIRouter()


async def process_and_notify(docs, upload_id):
    from starlette.concurrency import run_in_threadpool

    await run_in_threadpool(insert_documents, docs)
    await manager.send(upload_id, "completed")


@router.post("/upload_pdf", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        docs = await process_pdf(file)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    background_tasks.add_task(process_and_notify, docs, upload_id)
    return {"message": f"'{file.filename}' recibido. Ingesta programada."}
