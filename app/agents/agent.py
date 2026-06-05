from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.services.llm_service import LLMService
from app.rag.retriever import search_documents
from langsmith import traceable


llm = LLMService()

class AgentState(TypedDict):
     query: str
     decision: str
     result: str
     history: list

# Decide Action
@traceable(name="Decision Node")
def decide_tool(state: AgentState): 
    query = state["query"]
    history = state.get("history", [])

    prompt = f"""
    Conversation history:
    {history}
     Classify the user query into one of the categories:
    - "rag" → if question is about company policy or documents
    - "api" → if question is about live data like transactions, users

     Query: {query}
     
     Answer only "rag" or "api"
     """

    decision = llm.generate_response(prompt).strip().lower()
    return {"decision": decision}

# step 2 Rag path
@traceable(name="RAG Node")
def run_rag(state: AgentState):
    query = state["query"]
    history = state.get("history", [])

    docs = search_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Conversation history:
    {history}

    Answer using the context below:

    {context}

    Question: {query}
    
    Answer clearly.
    """

    answer = llm.generate_response(prompt)
    uodate_history = history + [f"query: {query}", f"answer: {answer}"]
    return {"result": answer,
            "history": uodate_history
            }


# step 2B: API path
@traceable(name="API Node")
def run_api(state: AgentState):
    query = state["query"]
    history = state.get("history", [])
    #mock API response
    if "failed" in query.lower():
        answer = "There are 5 failed transactions in the last 1 hour."
    else:
        answer = "API data not available."

    updated_history = history + [
        {"query": query, "answer": answer}
    ]

    return {
        "result": answer,
        "history": updated_history
    }

# step 3: Router
def route(state: AgentState):
    if state['decision'] == "api":
        return "api"
    return "rag"


# Build Graph
builder = StateGraph(AgentState)
builder.add_node("decide", decide_tool)
builder.add_node("rag", run_rag)
builder.add_node("api", run_api)

builder.set_entry_point("decide")
builder.add_conditional_edges(
    "decide",
    route,
    {
        "rag": "rag",
        "api": "api"
    }
)
builder.add_edge("rag", END)
builder.add_edge("api", END)

agent = builder.compile()
