# RetrievalQA chain

# Retriever
"""
    A retriever is an object that can take in a string and return some relevant documents

    To be "Retriever" the object must have a method called. "get_relevant_dcouments" that takes a string and return a list of documents
"""

# chain_type
"""
    chain_type="stuff"
        Take some context from the vectore store and "stuff" it into the prompt   
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("What is an interesting fact about English language ?")

print(result)
