from langchain_openai.chat_models import ChatOpenAI
from config import OPENROUTER_MODEL, OPENROUTER_BASE_URL, OPENROUTER_API_KEY

llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    openai_api_base=OPENROUTER_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
)
