"""
inference.py — BIS Standards Recommendation Engine
Entry Point (RAG / LLM Reranking Architecture)

Pipeline:
    1. Local Hybrid Search (FAISS + BM25) -> Top 15 Candidates
    2. Context Assembly -> Pass Top 15 standard chunks to Groq LLM
    3. LLM Reranker -> Outputs exact matching Standard IDs.
"""


import argparse
import json
import os
import re
import sys
import time
import pickle
import string
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")

PDF_PATH       = os.environ.get("BIS_PDF_PATH", "dataset.pdf")
INDEX_PATH     = "bis_index.faiss"
CHUNKS_PATH    = "bis_chunks.pkl"

EMBED_MODEL    = "./bis-finetuned-minilm" 

TOP_K_RETRIEVE = 40
TOP_K_RERANK   = 15
TOP_K_RETURN   = 5
RRF_K          = 60

FAISS_WEIGHT   = 1.0
BM25_WEIGHT    = 1.5

# ── Stopwords & Tokenization ──────────────────────────────────────────────────

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "i", "we", "you",
    "my", "our", "of", "in", "to", "for", "with", "on", "at", "from",
    "that", "which", "what", "this", "not", "no", "need", "want",
    "looking", "know", "should", "about", "their", "some", "any",
    "do", "does", "did", "have", "has", "had", "be", "been", "being",
    "it", "its", "he", "she", "they", "them", "his", "her",
    "would", "could", "will", "shall", "may", "might", "must", "can",
    "but", "or", "and", "if", "then", "so", "just", "also", "very",
    "there", "here", "where", "when", "how", "why", "who", "whom",
    "am", "im", "me", "us", "your", "those", "these", "than",
})

def tokenize(text: str) -> list:
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

# ── Step 1: Extraction & Chunking ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    print(f"  Reading PDF: {pdf_path}")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            if i % 50 == 0:
                print(f"  Pages processed: {i}/{total}...")
            full_text += (page.extract_text(x_tolerance=3, y_tolerance=3) or "") + "\n"
    return full_text

def extract_scope(chunk_text: str) -> str:
    patterns = [
        r'(?:1[.\s]+Scope|SCOPE)\s*[—\-:.]\s*(.+?)(?:\n\s*(?:2[.\s]|1\.\d|Note))',
        r'(?:1[.\s]+Scope|SCOPE)\s*[—\-:.]\s*(.+?)(?:\n\n)',
        r'(?:Scope)\s*[—\-:.]\s*(.+?)(?:\.\s)',
    ]
    for pat in patterns:
        m = re.search(pat, chunk_text, re.DOTALL | re.IGNORECASE)
        if m:
            scope = m.group(1).strip()
            return re.sub(r'\s+', ' ', scope)[:500]
    return ""

def extract_applications(chunk_text: str) -> str:
    patterns = [
        r'(?:used\s+(?:for|in))\s+(.+?)(?:\.|$)',
        r'(?:suitable\s+for)\s+(.+?)(?:\.|$)',
        r'(?:intended\s+for)\s+(.+?)(?:\.|$)',
        r'(?:covers?)\s+(.+?)(?:\.|$)',
        r'(?:application|use|purpose)\s*[:\-—]\s*(.+?)(?:\.|$)',
    ]
    apps = []
    for pat in patterns:
        for m in re.finditer(pat, chunk_text, re.IGNORECASE | re.MULTILINE):
            text = re.sub(r'\s+', ' ', m.group(1).strip())
            if len(text) > 10:
                apps.append(text[:200])
    seen = set()
    unique = []
    for a in apps:
        key = a.lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return " | ".join(unique[:5])

def chunk_by_standard(full_text: str) -> list:
    raw_chunks = re.split(r'SUMMARY\s+OF\s*\n+', full_text)
    raw_chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 100]
    
    is_pattern = re.compile(r'^IS\s+(\d+(?:\s*\(Part\s*\d+\))?)\s*[:\-]\s*(\d{4})\s+(.+?)(?:\n)', re.I)
    
    documents = []
    for chunk in raw_chunks:
        match = is_pattern.match(chunk)
        if match:
            sid = f"IS {match.group(1).strip()}: {match.group(2).strip()}"
            title = match.group(3).strip()
        else:
            sid = "Unknown"
            title = chunk.split('\n')[0][:100]
        
        scope = extract_scope(chunk)
        apps = extract_applications(chunk)
        
        # Reduced chunk size for LLM context window limits
        enriched = f"Standard ID: {sid}\nTitle: {title}\nScope: {scope}\nApplications: {apps}\nSnippet: {chunk[:500]}"
        documents.append({"standard_id": sid, "title": title, "enriched_text": enriched})
        
    return documents

# ── Engine ────────────────────────────────────────────────────────────────────

class BISEngine:
    def __init__(self):
        print("[1/3] Initializing Groq Reranker...")
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        print(f"[2/3] Loading Local Embeddings ({EMBED_MODEL})...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        
        print("[3/3] Loading Vector & Lexical Indices...")
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f: 
                self.documents = pickle.load(f)
        else:
            print("  No cache found. Building local index from PDF...")
            text = extract_text_from_pdf(PDF_PATH)
            self.documents = chunk_by_standard(text)
            
            embs = self.embed_model.encode(
                [d["enriched_text"] for d in self.documents], 
                normalize_embeddings=True
            )
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(np.array(embs).astype("float32"))
            faiss.write_index(self.index, INDEX_PATH)
            with open(CHUNKS_PATH, "wb") as f: 
                pickle.dump(self.documents, f)
                
        self.bm25 = BM25Okapi([tokenize(d["enriched_text"]) for d in self.documents])
        print(f"\nEngine Ready: {len(self.documents)} standards indexed.\n")

    def _llm_rerank(self, query: str, candidates: list) -> list:
        """Sends the retrieved documents to Groq and asks it to rank the IDs."""
        context_blocks = []
        for i, doc in enumerate(candidates):
            sid = doc['standard_id']
            title = doc['title']
            scope = doc['enriched_text'].split('Scope: ')[-1].split('Applications: ')[0].strip()
            context_blocks.append(f"[{i+1}] ID: {sid} | Title: {title} | Scope: {scope[:200]}")
            
        context_str = "\n".join(context_blocks)
        
        prompt = f"""You are a Bureau of Indian Standards (BIS) ranking expert.
A user has provided a query describing a construction material. 
I am providing you with {len(candidates)} candidate standard documents retrieved from our database.

User Query: "{query}"

Candidates:
{context_str}

Your task is to identify the MOST relevant Standard IDs for the query from the candidates provided.
Output a comma-separated list of EXACT Standard IDs, ordered from most relevant to least relevant.
If multiple standards are equally relevant, include them. Limit to max 5 IDs.
DO NOT output any explanations, formatting, or extra text. ONLY the comma-separated IDs.

Example Output: IS 459: 1992, IS 1626: 1984
"""
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            output = completion.choices[0].message.content.strip()
            # Parse output
            ranked_ids = [sid.strip() for sid in output.split(',')]
            # Clean up and ensure they are valid IDs
            ranked_ids = [sid for sid in ranked_ids if sid.startswith("IS ")]
            return ranked_ids
        except Exception as e:
            print(f"  [LLM Rerank Error: {e}] Falling back to RRF...")
            return []

    def query(self, query_text: str) -> tuple:
        start_time = time.time()
        
        # 1. Local Hybrid Retrieval
        q_emb = self.embed_model.encode([query_text], normalize_embeddings=True).astype("float32")
        _, faiss_indices = self.index.search(q_emb, TOP_K_RETRIEVE)
        
        bm25_scores = self.bm25.get_scores(tokenize(query_text))
        bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVE]

        scores = {}
        for rank, idx in enumerate(faiss_indices[0]): 
            if idx >= 0:
                scores[int(idx)] = scores.get(int(idx), 0) + (FAISS_WEIGHT / (RRF_K + rank + 1))
        for rank, idx in enumerate(bm25_indices): 
            idx = int(idx)
            scores[idx] = scores.get(idx, 0) + (BM25_WEIGHT / (RRF_K + rank + 1))
            
        top_fused = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        candidates = [self.documents[idx] for idx in top_fused[:TOP_K_RERANK]]
        
        # 2. LLM Reranking (RAG)
        llm_ranked_ids = self._llm_rerank(query_text, candidates)
        
        # 3. Compile Final List (LLM first, then fallback to RRF)
        final_ids, seen = [], set()
        
        for sid in llm_ranked_ids:
            # Only accept IDs that were actually in the candidate list to prevent hallucination
            if sid not in seen and any(sid == c["standard_id"] for c in candidates):
                final_ids.append(sid)
                seen.add(sid)
                
        # Pad remaining spots with RRF results if LLM didn't return 5
        for doc in candidates:
            sid = doc["standard_id"]
            if sid not in seen and sid != "Unknown":
                final_ids.append(sid)
                seen.add(sid)
            if len(final_ids) == TOP_K_RETURN:
                break
                
        latency = round(time.time() - start_time, 3)
        return final_ids, latency

def main():
    parser = argparse.ArgumentParser(description="BIS RAG Inference")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(INDEX_PATH):
        print("Pre-flight: Index cache found. Loading...")

    with open(args.input, "r") as f: 
        queries = json.load(f)
        
    engine = BISEngine()
    final_output = []
    
    for i, item in enumerate(queries):
        query_text = item.get("query", "")
        print(f"[{i+1}/{len(queries)}] Searching: {query_text[:60]}...")
        
        std_ids, lat = engine.query(query_text)
        print(f"  -> {std_ids} ({lat}s)")
        
        final_output.append({
            "id": item.get("id"),
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": std_ids,
            "latency_seconds": lat
        })

    with open(args.output, "w") as f: 
        json.dump(final_output, f, indent=2)
        
    print(f"\nEvaluation Complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()