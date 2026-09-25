# Support Assistant (Module 3)

This is a small offline RAG assistant for Zepto policy questions. I built it to take a set of policy documents, embed them in ChromaDB, pull out the most relevant chunks, and return a structured JSON answer through FastAPI.

The default path is intentionally simple and deterministic: `MOCK_LLM=1`. That means I can run it locally without an API key, without signing up for anything, and without making an external LLM call just to test the flow. The embeddings and retrieval still happen for real in this mode.

## Setup

```bash
cd support_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

1. Build the vector store from `docs/`:

   ```bash
   python ingest.py
   ```

2. Start the API. The mock mode is already the default, so you do not need to set anything:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 7860
   ```

3. Hit the endpoint:

   ```bash
   curl -s -X POST http://127.0.0.1:7860/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "When do I get my refund?"}'
   ```

## How it is structured

The workflow is simple and it lines up cleanly with the files in this folder:

**1. Ingestion** — `ingest.py`, `load_documents()`. Each file in `docs/doc_0N.txt` is loaded as a single chunk. Since each document is already a short policy topic, chunking per file is enough and there is no need to split a policy statement mid-sentence.

**2. Embeddings** — `ingest.py`, `build_collection()`. Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2`, which is a local model and does not require an API. The embeddings are stored in a persistent ChromaDB collection called `zepto_support_docs` inside `support_assistant/chroma_store/`, using cosine similarity (`hnsw:space: cosine`). The chunk IDs correspond to the document IDs (`doc_01` through `doc_08`).

**3. Retrieval** — `graph.py`, `retrieve_top_chunks()`. This is called from the `retrieve_and_answer` node. The incoming query is embedded with the same MiniLM model and matched against the vector store for the top 3 relevant chunks. This part runs for real in both modes and does not need any LLM or API key.

**4. Generation** — handled in `graph.py` by `retrieve_and_answer` and `direct_answer`. Intent classification and answer generation are the stages that branch on `MOCK_LLM`:

- `MOCK_LLM=1` (default): the assistant returns a canned answer built from the top retrieved chunk. `direct_answer` returns the fixed message, "I can only answer questions about Zepto policies right now." No network calls happen here.
- `MOCK_LLM=0` (optional extension): `classify_intent` asks Groq to classify the query, and the answer nodes use the structured prompt from `prompt_template.build_prompt()`. Responses are validated against `AskResponse`; invalid JSON gets up to two corrective retries after the first attempt. This path needs a Groq API key and makes external requests; the default path does neither.

### LangGraph flow

`graph.py` sets up a `StateGraph` over the `SupportState` `TypedDict` (`query`, `intent`, `retrieved_chunks`, `answer`, `sources`, `confidence`). There are three nodes:

- `classify_intent` — a quick keyword-based check on the lowercased query (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`). It decides whether the query is a policy question or a general one.
- `retrieve_and_answer` — this is used for policy questions and does the actual ChromaDB retrieval plus the mock generation.
- `direct_answer` — this is used for general questions and returns the fallback response without retrieval.

The conditional edge `route_from_intent` sends the graph to either `retrieve_and_answer` or `direct_answer`, depending on the intent. Both terminal nodes end at `END`.

### Structured output

`graph.py` uses a Pydantic model called `AskResponse` to enforce the response schema:

```python
class AskResponse(BaseModel):
    answer: str
    sources: List[str]       # chunk/doc ids used; [] for general_question
    confidence: float        # 0-1; fixed at 1.0 in mock mode
```

`main.py` exposes a `POST /ask` endpoint in FastAPI. It accepts `{"query": str}` and returns a validated `AskResponse`.

## Example calls

I started the server with `uvicorn main:app --host 127.0.0.1 --port 7860` without changing `MOCK_LLM`, so it ran in the default offline mode. The JSON responses below are the actual outputs from that setup.

**Example 1 — policy question that triggers retrieval**

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "When do I get my refund?"}'
```

```json
{
    "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop",
    "sources": [
        "doc_02",
        "doc_06",
        "doc_05"
    ],
    "confidence": 1.0
}
```

The top source here is `doc_02`, which is the correct policy document.

**Example 2 — general question, no retrieval**

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

**Example 3 — another retrieval example for gift cards**

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long is a gift card valid for?"}'
```

```json
{
    "answer": "Based on the retrieved context: Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 year from the",
    "sources": [
        "doc_07",
        "doc_02",
        "doc_01"
    ],
    "confidence": 1.0
}
```

This lands on `doc_07`, which is the right source document.

I also checked the retrieval manually. For questions like "How long is a gift card valid?", "when do I get my refund", "what are the support hours", and "can I cancel my order", the top hit landed on the correct document each time.

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

The image builds the vector store during the image build with `python ingest.py`, so the container is ready to serve `POST /ask` as soon as it starts. I built the image locally and tested the running container in the default mock mode; the refund example returned a grounded response with `doc_02` among its sources.

## The `MOCK_LLM` toggle

`MOCK_LLM` defaults to `"1"`, and that is the fully offline path this module is built around. There is no signup, no API key, and no network call to an LLM provider. Retrieval stays local and unchanged when the optional real path is enabled; intent classification and answer generation use Groq instead of the deterministic mock responses.

To try the optional Groq free tier, create a key through Groq's console and put it in the local, by creating `support_assistant/.env` file:

```dotenv
GROQ_API_KEY=your-new-key
MOCK_LLM=0
```

Then start the API from this folder:

```bash
uvicorn main:app --host 127.0.0.1 --port 7860
```

The default model is `llama-3.3-70b-versatile`; set `GROQ_MODEL` in `.env` to use a different model. These settings affect only the optional LLM calls; document ingestion, embeddings, and retrieval remain local.
