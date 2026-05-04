import os
import sys
import streamlit as st
from groq import Groq

# Add root to path so we can import BISEngine from inference.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inference import BISEngine

# ── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIS Standards Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body {
        font-family: 'Inter', sans-serif;
    }
    .best-match-card {
        background: #ffffff;
        border-left: 5px solid #2c3e50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 24px;
        border-radius: 8px;
        margin-bottom: 24px;
    }
    .best-match-title {
        font-size: 1.5em;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 12px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    .stat-box {
        background: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        padding: 18px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 16px;
    }
    .ai-rationale {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 6px;
        border-left: 3px solid #0056b3;
        margin-top: 16px;
        font-size: 0.95em;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Engine ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing BIS Standards Index...")
def load_engine():
    return BISEngine()

engine = load_engine()
doc_map = {d['standard_id']: d for d in engine.documents}

def generate_rationale(query: str, doc: dict) -> str:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return "Groq API key not found in environment. Please add it to your .env file to see AI rationale."
    
    client = Groq(api_key=groq_key)
    snippet = doc.get('enriched_text', '').split('Snippet: ')[-1][:600]
    prompt = f"""You are a Bureau of Indian Standards (BIS) compliance expert.
A user asked: "{query}"
The top recommended standard is: {doc['standard_id']} - {doc['title']}

Context from standard:
{snippet}

In exactly 2 or 3 sentences, explain precisely why this specific standard is the correct match for their product. Be professional and direct."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Rationale generation failed: {str(e)}"

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("BIS Standards Recommendation Engine")
st.caption("Enterprise RAG System Powered by Hybrid Semantic Search and LLM Reranking")
st.divider()

EXAMPLE_QUERIES = [
    "We manufacture 33 Grade Ordinary Portland Cement for structural construction",
    "Our factory produces hollow and solid concrete masonry blocks for walls",
    "We supply coarse and fine aggregates from natural sources for concrete mixing",
    "Our company manufactures corrugated asbestos cement sheets for roofing",
]

with st.sidebar:
    st.header("Example Queries")
    for ex in EXAMPLE_QUERIES:
        if st.button(ex[:40] + "...", key=ex, use_container_width=True):
            st.session_state["query_input"] = ex
            st.rerun()

query = st.text_area(
    "Describe your product or manufacturing process:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g. We manufacture 33 grade ordinary Portland cement for structural construction...",
)

search_btn = st.button("Find Standards", type="primary")

# ── Results ───────────────────────────────────────────────────────────────────

if search_btn and query.strip():
    with st.spinner("Processing query across vector and lexical indices..."):
        standard_ids, latency = engine.query(query)
        top_docs = [doc_map[sid] for sid in standard_ids if sid in doc_map]

    if not top_docs:
        st.error("No standards found.")
    else:
        col_main, col_stats = st.columns([3, 1])

        with col_stats:
            st.markdown("### Execution Analytics")
            st.markdown(f"""
            <div class="stat-box">
                <h2 style="margin:0; color:#2c3e50;">{latency}s</h2>
                <p style="margin:0; color:#6c757d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Inference Latency</p>
            </div>
            <div class="stat-box">
                <h2 style="margin:0; color:#2c3e50;">{len(top_docs)}</h2>
                <p style="margin:0; color:#6c757d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Standards Retrieved</p>
            </div>
            <div class="stat-box">
                <h2 style="margin:0; color:#2c3e50;">{len(engine.documents)}</h2>
                <p style="margin:0; color:#6c757d; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Total Index Size</p>
            </div>
            """, unsafe_allow_html=True)

        with col_main:
            st.markdown("### Primary Recommendation")
            best_doc = top_docs[0]
            
            with st.spinner("Generating AI compliance rationale..."):
                rationale = generate_rationale(query, best_doc)
            
            enriched = best_doc.get('enriched_text', '')
            scope = enriched.split('Scope: ')[-1].split('Applications: ')[0].strip() if 'Scope: ' in enriched else "Scope data unavailable."
            
            st.markdown(f"""
            <div class="best-match-card">
                <div class="best-match-title">{best_doc['standard_id']} — {best_doc['title']}</div>
                <div style="margin-bottom: 12px;"><b>Scope Summary:</b> {scope}</div>
                <div class="ai-rationale">
                    <strong>AI Rationale:</strong><br>
                    {rationale}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if len(top_docs) > 1:
                st.markdown("### Alternative Similar Standards")
                for doc in top_docs[1:]:
                    with st.expander(f"{doc['standard_id']} — {doc['title']}"):
                        d_enriched = doc.get('enriched_text', '')
                        d_scope = d_enriched.split('Scope: ')[-1].split('Applications: ')[0].strip() if 'Scope: ' in d_enriched else "Scope data unavailable."
                        d_snippet = d_enriched.split('Snippet: ')[-1].strip() if 'Snippet: ' in d_enriched else doc.get('raw_text', '')[:400]
                        
                        st.markdown(f"**Scope Summary:** {d_scope}")
                        st.markdown(f"**Context Snippet:** {d_snippet[:500]}...")

elif search_btn:
    st.warning("Please enter a product description.")