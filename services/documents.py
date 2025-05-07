import pdfplumber
from io import BytesIO
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime


async def process_pdf(file: UploadFile, upload_id: str):
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Solo se permiten PDFs.")

    pdf_bytes = await file.read()
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Envolver el texto como un documento con metadata
    metadata = {
        "filename": file.filename,
        "upload_id": upload_id,
        "timestamp": datetime.utcnow().isoformat(),
    }

    documents = [Document(page_content=text, metadata=metadata)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Propagar metadata a los fragmentos (ya que Langchain puede no copiarla)
    for doc in chunks:
        doc.metadata = metadata.copy()

    return chunks
