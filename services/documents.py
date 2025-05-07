import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile

async def process_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Solo se permiten PDFs.")
    path = os.path.basename(file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    pages = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)
