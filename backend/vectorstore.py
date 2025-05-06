import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

INDEX_PATH = "faiss_index"

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
