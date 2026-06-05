from fastapi import FastAPI
from app.services.llm_service import LLMService
from app.rag.retriever import search_documents
from app.agents.agent import agent

app = FastAPI()
llm_service = LLMService()
 

@app.get("/")
def root():
    return {"message":"Enterprise AI copilot Running"}



@app.get("/test-llm")
def test_llm():
    response = llm_service.generate_response("Explain AI agents in simple terms")
    return {"response": response}

@app.get("/search")
def search(query: str):
    docs = search_documents(query)
    return {"results": [doc.page_content for doc in docs]}


@app.get("/ask")
def ask(query: str):
    docs = search_documents(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}

    Return a complete sentence.
    If not found, say "Not found".
    """

    response = llm_service.generate_response(prompt)
    return {"answer": response}

conversation_store = {}

@app.get("/agent")
def run_agent(query: str, session_id: str = "default"):
    history = conversation_store.get(session_id, [])
    result = agent.invoke({"query": query,
                           "history": history})
    conversation_store[session_id] = result.get("history",[])
    return {"response": result["result"],
            "history": conversation_store[session_id]
            }