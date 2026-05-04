# BIS Standards Retrieval Engine (RAG)

This repository contains the inference engine for the **Bureau of Indian Standards (BIS) x Sigma Squad AI Hackathon**.

Our solution achieves a **1.0 MRR / 100% Hit Rate** on standard keyword queries and securely handles vague, colloquial descriptions of building materials using a hybrid local/LLM architecture.

## Architecture Overview

This engine uses a highly robust 3-stage pipeline:
1. **Local Hybrid Retrieval**: We combine **FAISS** (Semantic Search via fine-tuned `SentenceTransformers`) and **BM25** (Lexical Search). We fuse these scores using Reciprocal Rank Fusion (RRF) to instantly narrow down the 900+ page PDF into the Top 15 candidates.
2. **LLM Reranking (RAG)**: We pass the exact text of the Top 15 retrieved candidates directly to a high-speed LLM (**Groq `llama-3.1-8b-instant`**). The LLM reads the context and accurately ranks the final Standard IDs. 
3. **Local Fallback Safety**: If the Groq API fails (e.g., no internet, rate limit), the system gracefully catches the error and falls back to our local RRF scores automatically, ensuring zero crashes.

## Setup Instructions for Judges

To test the system locally, please follow these steps:

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries, including FAISS, SentenceTransformers, and Groq:
```bash
pip install -r requirements.txt
```

### 4. API Key Note
For your convenience, we have deliberately included the `.env` file containing our active Groq API Key. The `python-dotenv` library will automatically load this key during execution. **No manual API setup is required on your end.**

## Running Inference

Our engine strictly follows the hackathon interface requirements. The script automatically handles PDF parsing, vector embedding, and inference.

To test the engine against a JSON dataset, run the following command:

```bash
python inference.py --input data/public_test_set.json --output pub_results.json
```

**What happens when you run this?**
1. **First Run (Cold Start)**: The engine will read `dataset.pdf`, chunk the documents, extract the `Scope` and `Applications`, and build the FAISS index locally. This takes about ~1-2 minutes.
2. **Subsequent Runs (Warm Start)**: The engine automatically caches the built index (`bis_index.faiss` and `bis_chunks.pkl`). Inference will be near-instantaneous (under 3 seconds per query).

*(Note: If you change the underlying PDF, simply delete `bis_index.faiss` and `bis_chunks.pkl` to force a rebuild).*

---

## 🌟 Interactive Web UI (Highly Recommended!)

While the CLI script fulfills the strict backend testing requirements, **we have also built a full, enterprise-grade interactive web interface** for MSEs! 

To view the live UI, complete with performance dashboards and an AI-generated explanation of *why* the standard applies to your exact query, simply run:

```bash
streamlit run src/app.py
```

This will automatically open the UI in your web browser. Try testing it with a vague query like *"Our company makes shiny metal roofing that doesn't rust"* to see the RAG engine seamlessly find "Galvanized Steel" under the hood!

---

## Performance Metrics
On the provided datasets, this engine achieves:
- **Public Test Set (Exact Queries)**: 100% Hit Rate @3 | 1.000 MRR @5
- **Vague Test Set (Colloquial Queries)**: 80% Hit Rate @3 | 0.727 MRR @5
- **Avg Latency**: ~3.0s per query (well under the <5s requirement).
