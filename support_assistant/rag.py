import os
from pathlib import Path
from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
try:
    from .prompt import build_policy_prompt
except ImportError:
    from prompt import build_policy_prompt

BASE = Path(__file__).parent
DOCS = BASE / "docs"

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list[dict]
    response: dict

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(BASE / "chroma_db"))
collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

def ingest():
    existing = collection.count()
    if existing >= 8:
        return
    ids, texts, metas = [], [], []
    for p in sorted(DOCS.glob("doc_*.txt")):
        text = p.read_text(encoding="utf-8")
        ids.append(p.stem)
        texts.append(text)
        metas.append({"document_id": p.stem})
    embeddings = model.encode(texts).tolist()
    collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)

def classify_intent(state: GraphState):
    q = state["query"].lower()
    keywords = [
        "delivery", "return", "refund", "membership",
        "tracking", "cancel", "gift card", "support hours"
    ]
    intent = "policy_question" if any(k in q for k in keywords) else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState):
    q = state["query"]
    q_embedding = model.encode([q]).tolist()
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=3
    )
    docs = result["documents"][0]
    ids = result["ids"][0]
    retrieved = [
        {"id": i, "text": d}
        for i, d in zip(ids, docs)
    ]
    context = "\n\n".join(f"[{x["id"]}] {x["text"]}" for x in retrieved)
    prompt = build_policy_prompt(q, context)
    _ = prompt  # Structured prompt is part of the deterministic mock execution path.
    top = retrieved[0]["text"]
    snippet = top[:200]
    response = AnswerResponse(
        answer=f"Based on the retrieved context: {snippet}",
        sources=[x["id"] for x in retrieved],
        confidence=1.0
    )
    return {"retrieved": retrieved, "response": response.model_dump()}

def direct_answer(state: GraphState):
    response = AnswerResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return {"response": response.model_dump()}

def route(state: GraphState):
    return state["intent"]

def build_graph():
    ingest()
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer"
        }
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()

app_graph = build_graph()

def ask(query: str) -> AnswerResponse:
    result = app_graph.invoke({"query": query})
    return AnswerResponse.model_validate(result["response"])

if __name__ == "__main__":
    print(ask("What is the delivery fee?").model_dump_json())
    print(ask("What is your favorite color?").model_dump_json())
