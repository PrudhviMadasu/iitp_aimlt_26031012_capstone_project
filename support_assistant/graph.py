"""
This is the LangGraph flow for the support assistant.

The flow looks like this:
    classify_intent -> (conditional edge) -> retrieve_and_answer -> END
                                           -> direct_answer       -> END

`classify_intent` is just a keyword-based classifier. It does not use an LLM
and behaves the same in both mock modes.

`retrieve_and_answer` always does a real ChromaDB lookup with embedding and
cosine similarity. The generation step inside that node changes depending on
`MOCK_LLM`:
  - `MOCK_LLM=1` (default): return a simple answer built from the top
    retrieved chunk with no network call.
  - `MOCK_LLM=0`: would call a real LLM using the prompt from
    `prompt_template.py`. That extension is not implemented here and raises
    `NotImplementedError`.

`direct_answer` handles general questions that are not about Zepto policies.
It also uses the mock path by default and raises the same error in the real
LLM path.
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import TypedDict, List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END

from ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from prompt_template import build_prompt

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).with_name(".env"))

# Quick keyword list for the policy-question heuristic. It checks the
# lowercased query and routes it to the policy or general path.
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

TOP_K = 3
SNIPPET_LEN = 200
MOCK_CONFIDENCE = 1.0
GENERAL_QUESTION_FALLBACK = (
    "I can only answer questions about Zepto policies right now."
)

# --- Shared resources that are initialized only when needed ---

_embedding_model = None
_collection = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def is_mock_mode() -> bool:
    """MOCK_LLM defaults to '1', which keeps this fully offline by default."""
    return os.environ.get("MOCK_LLM", "1") == "1"


# --- Structured output schema (Pydantic) ---


class AskResponse(BaseModel):
    answer: str = Field(min_length=1)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# --- Graph state ---


class SupportState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: List[dict]  # [{"id": str, "text": str, "distance": float}]
    answer: str
    sources: List[str]
    confidence: float


# --- Nodes ---


def _call_groq_json(prompt: str, correction: str = "") -> str:
    """Call Groq's OpenAI-compatible endpoint and return its raw JSON text."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY to use the optional MOCK_LLM=0 path.")

    messages = [{"role": "user", "content": prompt}]
    if correction:
        messages.append({"role": "user", "content": correction})

    payload = {
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc
    return result["choices"][0]["message"]["content"]


def _classify_with_llm(query: str) -> str:
    prompt = (
        "Classify this user query as JSON with exactly one key, intent. "
        "The value must be policy_question or general_question. A question "
        "about Zepto delivery, returns, refunds, membership, tracking, "
        "cancellation, gift cards, or support hours is policy_question. "
        f"Query: {query}"
    )
    correction = ""
    for attempt in range(3):
        try:
            result = json.loads(_call_groq_json(prompt, correction))
            intent = result["intent"]
            if intent not in {"policy_question", "general_question"}:
                raise ValueError("intent must be policy_question or general_question")
            return intent
        except (ValueError, KeyError, TypeError) as exc:
            correction = (
                "Your previous response was invalid: " + str(exc)
                + ". Return only a JSON object with intent set to "
                "policy_question or general_question."
            )
            if attempt == 2:
                raise RuntimeError("Intent classification failed after 3 attempts.") from exc
    raise RuntimeError("Intent classification failed.")


def classify_intent(state: SupportState) -> SupportState:
    """Use the required keyword heuristic by default, or the optional LLM path."""
    query = state["query"]
    if is_mock_mode():
        query_lower = query.lower()
        intent = (
            "policy_question"
            if any(keyword in query_lower for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        intent = _classify_with_llm(query)
    return {"intent": intent}


def retrieve_top_chunks(query: str, top_k: int = TOP_K) -> List[dict]:
    """Embed the query and pull the top_k most relevant chunks from ChromaDB."""
    model = _get_embedding_model()
    collection = _get_collection()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    for chunk_id, text, distance in zip(ids, documents, distances):
        chunks.append({"id": chunk_id, "text": text, "distance": distance})
    return chunks


def _generate_with_retry(query: str, chunks: List[dict], mode: str) -> AskResponse:
    """Generate and validate a structured answer, retrying invalid responses."""
    prompt = build_prompt(query, chunks)
    allowed_sources = [chunk["id"] for chunk in chunks]
    correction = ""
    for attempt in range(3):
        try:
            raw_response = _call_groq_json(prompt, correction)
            response = AskResponse.model_validate_json(raw_response)
            if not set(response.sources).issubset(allowed_sources):
                raise ValueError("response sources must come from retrieved document IDs")
            return response.model_copy(update={"sources": allowed_sources})
        except (ValidationError, ValueError) as exc:
            correction = (
                "Your previous response failed validation: " + str(exc)
                + ". Return only a JSON object with non-empty string answer, "
                "sources as an array of supplied document IDs, and confidence "
                "as a number from 0 to 1. Do not include markdown fences."
            )
            if attempt == 2:
                return AskResponse(
                    answer=(
                        f"[ERROR] Real-LLM response for {mode} failed validation "
                        "after 3 attempts."
                    ),
                    sources=allowed_sources,
                    confidence=0.0,
                )
    raise RuntimeError(f"Real-LLM response for {mode} failed.")


def retrieve_and_answer(state: SupportState) -> SupportState:
    chunks = retrieve_top_chunks(state["query"], top_k=TOP_K)
    sources = [c["id"] for c in chunks]

    if is_mock_mode():
        top_chunk_snippet = chunks[0]["text"][:SNIPPET_LEN] if chunks else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
    else:
        answer = _generate_with_retry(
            state["query"], chunks, "retrieve_and_answer"
        ).answer

    return {
        "retrieved_chunks": chunks,
        "answer": answer,
        "sources": sources,
        "confidence": MOCK_CONFIDENCE,
    }


def direct_answer(state: SupportState) -> SupportState:
    if is_mock_mode():
        answer = GENERAL_QUESTION_FALLBACK
    else:
        answer = _generate_with_retry(state["query"], [], "direct_answer").answer

    return {
        "retrieved_chunks": [],
        "answer": answer,
        "sources": [],
        "confidence": MOCK_CONFIDENCE,
    }


def route_from_intent(state: SupportState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AskResponse:
    """Run the graph for one query and return the structured response."""
    app = get_compiled_graph()
    result = app.invoke({"query": query})
    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", MOCK_CONFIDENCE),
    )
