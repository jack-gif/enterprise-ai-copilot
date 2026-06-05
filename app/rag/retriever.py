from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from app.services.embedding_service import  GeminiEmbeddings


from langchain_community.vectorstores import FAISS
from app.services.embedding_service import GeminiEmbeddings
from langsmith import traceable


def get_vectorstore():
    embeddings = GeminiEmbeddings()

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

@traceable(name="Vector Search")
def search_documents(query: str):
    vectorstore = get_vectorstore()

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    # Filter relevant docs
    filtered_docs = [
        doc for doc, score in docs_with_scores if score < 0.75
    ]

    # fallback if nothing matched
    if not filtered_docs:
        filtered_docs = [doc for doc, _ in docs_with_scores[:1]]

    for doc, score in docs_with_scores:
        print("SCORE:", score)
        print(doc.page_content)

    return filtered_docs