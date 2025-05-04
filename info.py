import os
import logging
import asyncio
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Form,
)
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

# ── Load env variables ────────────────────────────────────────────────────────
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TEMP_DIR = "temp"
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

# ── WebSocket Conexiones ───────────────────────────────────────────────────────
active_connections: Dict[str, WebSocket] = {}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # Mantener la conexión viva
    except WebSocketDisconnect:
        active_connections.pop(client_id, None)


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


class AskRequest(BaseModel):
    question: str


async def send_ws_message(client_id: str, message: str):
    websocket = active_connections.get(client_id)
    if websocket:
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"⚠️ Error enviando mensaje WS a {client_id}: {e}")


def insert_documents(documents: List[Document], client_id: str, loop):
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

    # Notificar al cliente por WebSocket al terminar
    future = asyncio.run_coroutine_threadsafe(
        send_ws_message(client_id, "✅ PDF procesado e insertado en Qdrant."), loop
    )


@app.post("/upload_pdf", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    client_id: str = Form(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten PDFs.")

    # Guardar PDF
    pdf_path = os.path.join(TEMP_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Cargar y trocear
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    loop = asyncio.get_event_loop()
    background_tasks.add_task(insert_documents, docs, client_id, loop)

    return {"message": f"'{file.filename}' recibido. Ingesta en Qdrant programada."}


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
