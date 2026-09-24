"""
Simple FastAPI wrapper around the LangGraph support assistant.

Run it like this:
    uvicorn main:app --host 0.0.0.0 --port 7860

This expects the vector store to already exist in chroma_store/, so run
`python ingest.py` once before starting the API.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from graph import run_query, AskResponse

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}
