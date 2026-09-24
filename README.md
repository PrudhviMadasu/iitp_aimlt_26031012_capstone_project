# Zepto Data & AI Platform

This project is split into three modules, and each one has its own dependency set. I kept a separate `requirements.txt` inside each module instead of using one consolidated file, because the data pipeline, analytics, and support assistant all rely on different libraries and version ranges.

## Setup

From the repo root:

```bash
# data_pipeline
cd data_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..

# analytics
cd analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..

# support_assistant
cd support_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

This keeps the Python environments isolated and avoids dependency conflicts between the notebook-based analytics stack and the local RAG stack.

## Run each module end to end

### 1) data_pipeline

```bash
cd data_pipeline
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace data_pipeline.ipynb
```

This module scrapes book listings, cleans the data, converts prices from GBP to INR using a fixed rate, builds a normalized SQLite database, and runs both SQL and pandas validation checks. The output is stored in `books.db`, and the notebook is the execution entry point.

### 2) analytics

```bash
cd analytics
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

I run the EDA notebook first so it creates the cleaned `titanic.csv` file. Then I run the modeling notebook to train and compare models, tune the selected classifier, and save the final pipeline artifact as `titanic_pipeline.joblib`.

### 3) support_assistant

```bash
cd support_assistant
source .venv/bin/activate
python ingest.py
uvicorn main:app --host 127.0.0.1 --port 7860
```

Then in a second terminal:

```bash
curl -s -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "When do I get my refund?"}'
```

This builds the local vector store from the policy documents and exposes a FastAPI endpoint for asking policy questions. The project uses the default offline mode, so no external LLM API key is required.

## Design decisions by module

### data_pipeline

I built this as a small ETL pipeline: scrape category pages, clean the product fields, normalize the schema, and store the data in SQLite. A two-table structure with `categories` and `books` keeps the database realistic and avoids repeating category names in every row.

### analytics

The analytics module is split into two notebooks so the EDA stays separate from the modeling work. The EDA notebook produces a cleaned dataset that the modeling notebook reuses, which keeps the workflow reproducible and avoids reloading raw data. I ended up favoring Logistic Regression because it is simpler and generalizes better on this dataset than the more complex tree-based models.

### support_assistant

The support assistant uses local embeddings plus vector retrieval over the policy documents, with a FastAPI endpoint on top. The design is intentionally offline-first: it stores the embedded documents in ChromaDB locally and uses a mock generation path by default so the assistant can be run and tested without depending on a paid external LLM service.

