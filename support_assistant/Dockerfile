FROM python:3.9-slim

WORKDIR /app

# This is needed for ChromaDB's hnswlib native extension.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY docs/ ./docs/
COPY ingest.py prompt_template.py graph.py main.py ./

# Build the vector store during image creation so the container is ready to
# serve as soon as it starts. This keeps the runtime simple and avoids extra
# setup after `docker run`.
RUN python ingest.py

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
