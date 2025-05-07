from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, ask
from sse import manager
from config import startup_check
from jobs.cleanup import cleanup_job
import asyncio

app = FastAPI(title="RAG con Qdrant + OpenRouter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await startup_check()
    asyncio.create_task(cleanup_job())


app.include_router(upload.router)
app.include_router(ask.router)
app.include_router(manager.router)
