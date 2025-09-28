# RAG_app.py — complete RAG app (Steps 3.1–3.10)
# Works from repo root or inside a subfolder. Uses a local extractive fallback
# so you still get answers if the OpenAI call is blocked or returns nothing.

# --- Force Transformers to ignore TensorFlow/Keras (we use PyTorch) ---
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ---------------- 3.1 Suppress noisy logs ----------------
import logging
import warnings
import transformers.utils.logging as hf_logging

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ---------------- Common imports & path base ----------------
from pathlib import Path
import re
import numpy as np
import faiss

BASE_DIR = Path(__file__).resolve().parent  # script’s folder (root-agnostic paths)

# ---------------- 3.2 Load API key from .env ----------------
from dotenv import load_dotenv
import openai
from openai import BadRequestError

# ensure .env next to this file is loaded
load_dotenv(BASE_DIR / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------- 3.3 Parameters ----------------
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

# Re-ranking parameters
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

# ---------------- 3.4 Read the pre-scraped document ----------------
DOC_PATH = BASE_DIR / "Selected_Document.txt"
with open(DOC_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# ---------------- 3.5 Split into appropriately-sized chunks ----------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_text(text)
print(f"[INFO] Split document into {len(chunks)} chunks.")

# ---------------- 3.6 Embed & Build FAISS Index ----------------
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings_arr = np.asarray(embeddings, dtype=np.float32)

dim = embeddings_arr.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings_arr)
print(f"[INFO] Built FAISS index with {faiss_index.ntotal} vectors, dim={dim}.")

# ---------------- 3.7 Retrieval Function ----------------
def retrieve_chunks(question: str, k: int = top_k):
    """
    Encode the question, query FAISS for top-k nearest neighbors,
    and return the corresponding text chunks.

    Assumes: embedder, faiss_index, chunks already exist.
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.asarray(q_vec, dtype=np.float32)
    D, I = faiss_index.search(q_arr, k)
    return [chunks[i] for i in I[0] if i != -1]

# ---------------- 3.8 Implement a Cross-Encoder Re-Ranker ----------------
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(cross_encoder_name)  # higher score = more relevant

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for it in items:
        key = _normalize_ws(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    """
    Score top_k retrieved chunks against the question using the cross-encoder,
    then keep only the best top_m for the final context.

    NOTE: No bi-encoder re-encoding here; this only uses the cross-encoder for scoring.
    """
    if not candidate_chunks:
        return []
    pairs = [(question, _normalize_ws(c)) for c in candidate_chunks]
    scores = reranker.predict(pairs)  # higher = more relevant
    ranked = sorted(zip(scores, candidate_chunks), key=lambda t: t[0], reverse=True)
    selected = [c for _, c in ranked[:m]]
    return dedupe_preserve_order(selected)

# ---------------- 3.9 Q&A with ChatGPT (with robust fallbacks) ----------------
def _extractive_fallback(context: str, question: str) -> str:
    """
    Zero-cost fallback: pick the most relevant sentences from context if the API fails.
    """
    sents = re.split(r'(?<=[.!?])\s+', context)
    kw = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", question)}
    scored = []
    for s in sents:
        hits = sum(1 for w in kw if w in s.lower())
        if hits:
            scored.append((hits, s))
    if not scored:
        return "I don’t know based on the provided context."
    scored.sort(key=lambda t: t[0], reverse=True)
    return " ".join(s for _, s in scored[:4]).strip()

def _chat_completion(system_prompt: str, user_prompt: str) -> str:
    """
    Call Chat Completions with conservative params and multiple fallbacks:
      - swap between max_tokens / max_completion_tokens
      - avoid temperature when the model disallows it
      - try gpt-5-mini first, then gpt-5
    Returns '' on failure so caller can use the extractive fallback.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    attempts = [
        ("gpt-5-mini", {"max_completion_tokens": 500}),
        ("gpt-5-mini", {"max_tokens": 500}),
        ("gpt-5",      {"max_completion_tokens": 500}),
        ("gpt-5",      {"max_tokens": 500}),
    ]
    for model, kwargs in attempts:
        try:
            resp = openai.chat.completions.create(model=model, messages=messages, **kwargs)
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                return txt
        except (BadRequestError, Exception):
            continue
    return ""

def answer_question(question: str) -> str:
    """
    1) Retrieve candidate chunks (top_k = 20).
    2) Re-rank with cross-encoder and keep top_m.
    3) Build context and query ChatGPT.
    4) If API gives nothing, fall back to extractive snippets from the context.
    """
    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)
    context = "\n\n".join(relevant_chunks)

    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don’t know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    reply = _chat_completion(system_prompt, user_prompt)
    if not reply:
        reply = _extractive_fallback(context, question)
    return reply

# ---------------- 3.10 Interactive loop ----------------
if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        try:
            question = input("Your question: ")
        except EOFError:
            break
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
