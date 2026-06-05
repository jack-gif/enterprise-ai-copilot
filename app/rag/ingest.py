from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.services.embedding_service import GeminiEmbeddings
import os
import pickle


 


def ingest_documents():
    with open('app/rag/data/company_policy.txt','r', encoding='utf-8') as file:
        text = file.read()

        docs = [Document(page_content=text)]

        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)

        split_docs = splitter.split_documents(docs)

        embeddings = GeminiEmbeddings()

        vectorstore = FAISS.from_documents(split_docs, embeddings)

        vectorstore.save_local('vectorstore')


if __name__ == "__main__":
    ingest_documents()