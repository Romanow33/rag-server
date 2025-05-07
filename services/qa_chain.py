from services.qdrant import client, COLLECTION_NAME
from services.embeddings import embedding
from services.llm import llm
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from langchain.chains import RetrievalQA

# Se construye una vez, al importar este módulo
vectorstore = LangchainQdrant(
    client=client, collection_name=COLLECTION_NAME, embeddings=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limita resultados
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # Opcionalmente puedes probar también "refine"
    retriever=retriever,
)
