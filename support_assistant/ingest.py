"""
This is the ingestion and embedding step.

It reads the policy docs from docs/, treats each file as one chunk, embeds
those chunks with sentence-transformers/all-MiniLM-L6-v2, and stores them
in a local ChromaDB collection. Each policy file is already a short,
single-topic document, so one chunk per file is enough and keeps things
simple.

Run this once before starting the API:

    python ingest.py
"""

import os
import glob

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "zepto_support_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Load each doc_XX.txt as one chunk. Returns [{"id", "text", "source"}]."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text, "source": os.path.basename(path)})
    return chunks


def build_collection(
    docs_dir: str = DOCS_DIR,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
):
    """Embed all document chunks and (re)build the ChromaDB collection."""
    chunks = load_documents(docs_dir)
    if not chunks:
        raise RuntimeError(f"No doc_*.txt files found in {docs_dir}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode([c["text"] for c in chunks]).tolist()

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    # Rebuild the collection each time so rerunning ingest.py is predictable.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )

    return collection


if __name__ == "__main__":
    collection = build_collection()
    print(f"Ingested {collection.count()} chunks into collection '{COLLECTION_NAME}'")
    print(f"Chroma store: {CHROMA_PATH}")
