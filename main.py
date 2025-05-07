from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, ask
from sse import manager
from config import TEMP_DIR, startup_check

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


app.include_router(upload.router)
app.include_router(ask.router)
app.include_router(manager.router)


""" import asyncio, os
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant as LangchainQdrant
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

# ── Load env variables ────────────────────────────────────────────────────────
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Inicializar servicios ─────────────────────────────────────────────────────
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

app = FastAPI(title="RAG con Qdrant + OpenRouter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔥 permite todos los orígenes
    allow_credentials=False,  # ⚠️ si lo pones en True, no puedes usar "*" en allow_origins
    allow_methods=["*"],  # permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # permite todas las cabeceras
)

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


# ——————————————————————————
# 1) ConnectionManager para SSE
# ——————————————————————————
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[asyncio.Queue]] = {}

    async def connect(self, upload_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.active.setdefault(upload_id, []).append(q)
        return q

    def disconnect(self, upload_id: str, q: asyncio.Queue):
        lst = self.active.get(upload_id, [])
        if q in lst:
            lst.remove(q)
        if not lst:
            self.active.pop(upload_id, None)

    async def send(self, upload_id: str, message: str):
        for q in self.active.get(upload_id, []):
            await q.put(message)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🔌 Inicializando conexión a Qdrant en %s …", QDRANT_URL)
        _ = qdrant_client.get_collections()
        logger.info("✅ Conexión a Qdrant exitosa.")
    except Exception as e:
        logger.error("❌ Error al conectar a Qdrant: %s", e, exc_info=True)

    if not os.path.exists(TEMP_DIR) or not os.access(TEMP_DIR, os.W_OK):
        raise RuntimeError(f"❌ No se puede escribir en {TEMP_DIR}.")

    print(f"✅ Directorio {TEMP_DIR} disponible y con permisos de escritura.")


def insert_documents(documents: List[Document]):
    if COLLECTION_NAME not in [
        c.name for c in qdrant_client.get_collections().collections
    ]:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vectorstore = LangchainQdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding,
    )
    vectorstore.add_documents(documents)


class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[asyncio.Queue]] = {}

    async def connect(self, upload_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.active.setdefault(upload_id, []).append(q)
        return q

    def disconnect(self, upload_id: str, q: asyncio.Queue):
        lst = self.active.get(upload_id, [])
        if q in lst:
            lst.remove(q)
        if not lst:
            self.active.pop(upload_id, None)

    async def send(self, upload_id: str, message: str):
        for q in self.active.get(upload_id, []):
            await q.put(message)


manager = ConnectionManager()


@app.get("/notifications/{upload_id}")
async def notifications(upload_id: str):
    q = await manager.connect(upload_id)

    async def event_generator():
        try:
            while True:
                msg = await q.get()
                # enviamos el evento al cliente
                yield {"data": msg}
                # si es el mensaje de finalización, salimos del bucle
                if msg == "completed":
                    break
        except asyncio.CancelledError:
            # cliente se desconectó antes
            pass
        finally:
            # limpiar la suscripción
            manager.disconnect(upload_id, q)

    # ping=0 para no enmascarar eventos con comentarios
    return EventSourceResponse(event_generator(), ping=0)


async def process_and_notify(docs: List[Document], upload_id: str):
    # insert_documents corre en un hilo
    await run_in_threadpool(insert_documents, docs)
    # luego enviamos el evento
    await manager.send(upload_id, "completed")


@app.post("/upload_pdf", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se permiten PDFs.")
    pdf_path = os.path.join("/tmp", file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    # Una sola tarea: inserta y luego notifica
    background_tasks.add_task(process_and_notify, docs, upload_id)
    return {"message": f"'{file.filename}' recibido. Ingesta programada."}


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: AskRequest):
    vectorstore = LangchainQdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding,
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    answer = qa_chain.run(request.question)
    logger.info("❓ Pregunta: %s → respuesta generada", request.question)
    return {"question": request.question, "answer": answer}
 """
