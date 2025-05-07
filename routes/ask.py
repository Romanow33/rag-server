from fastapi import APIRouter
from pydantic import BaseModel
from services.qa_chain import qa_chain
from config import logger

router = APIRouter()

class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(request: AskRequest):
    answer = qa_chain.run(request.question)
    logger.info("❓ Pregunta: %s → respuesta generada", request.question)
    return {"question": request.question, "answer": answer}
