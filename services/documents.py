from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile
from tempfile import SpooledTemporaryFile


async def process_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Solo se permiten PDFs.")

    pdf_bytes = await file.read()

    with SpooledTemporaryFile() as tmp:
        tmp.write(pdf_bytes)
        tmp.seek(0)
        loader = PyPDFLoader(tmp)
        pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)
