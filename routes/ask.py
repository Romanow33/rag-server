from fastapi import APIRouter
from pydantic import BaseModel
from services.qdrant import client, COLLECTION_NAME
from services.embeddings import embedding
from services.llm import llm
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from langchain.chains import RetrievalQA
from config import logger

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(request: AskRequest):
    vectorstore = LangchainQdrant(
        client=client, collection_name=COLLECTION_NAME, embeddings=embedding
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    answer = qa_chain.run(request.question)
    logger.info("❓ Pregunta: %s → respuesta generada", request.question)
    return {"question": request.question, "answer": answer}
