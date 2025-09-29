# RAG_app.py
# Retrieval-Augmented Generation pipeline:
# - Suppress noisy logs
# - Load API key from .env
# - Parameters
# - Read Selected_Document.txt
# - Split into chunks
# - Embed with Sentence-Transformers
# - Build FAISS index
# - Retrieve + Cross-encode re-rank
# - ChatGPT synthesis (with safe fallbacks)
# - Interactive loop

from __future__ import annotations

import os
import re
import sys
import json
import time
import logging
import warnings
from typing import List, Tuple

# --------------------------- 3.1 Suppress noisy logs ---------------------------

# avoid TF/Keras import path issues from transformers on CPU-only boxes
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import logging as hf_logging  # type: ignore

warnings.filterwarnings("ignore")
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

# --------------------------- std / third-party imports -------------------------

import numpy as np  # type: ignore
import faiss  # type: ignore

from dotenv import load_dotenv  # python-dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore

import openai  # OpenAI SDK v1-compatible Chat Completions API

# --------------------------- 3.2 API credentials -------------------------------

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not openai.api_key:
    print("[WARN] OPENAI_API_KEY not found in .env; LLM synthesis will fall back to extractive mode.")

# --------------------------- 3.3 Parameters ------------------------------------

chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

# Re-ranking parameters
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

# --------------------------- 3.4 Read pre-scraped document ---------------------

DOC_PATH = os.path.join(os.path.dirname(__file__), "Selected_Document.txt")

with open(DOC_PATH, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

if not text.strip():
    print("[ERROR] Selected_Document.txt is empty. Run text_extractor.py first.")
    sys.exit(1)

# --------------------------- 3.5 Split into chunks -----------------------------

separators = ["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=separators,
)

chunks: List[str] = splitter.split_text(text)
print(f"[INFO] Split document into {len(chunks)} chunks.")

# --------------------------- 3.6 Embed + FAISS --------------------------------

embedder = SentenceTransformer(model_name)
emb_matrix = embedder.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=False,
)
if not isinstance(emb_matrix, np.ndarray):
    emb_matrix = np.asarray(emb_matrix)

emb_matrix = emb_matrix.astype("float32")
dim = emb_matrix.shape[1]

faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(emb_matrix)
print(f"[INFO] Built FAISS index with {faiss_index.ntotal} vectors, dim={dim}.")

# --------------------------- 3.7 Retrieval function ----------------------------

def retrieve_chunks(question: str, k: int = top_k) -> List[str]:
    """
    Encode the query, search FAISS for nearest neighbors, and return the top-k text chunks.
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.asarray(q_vec, dtype="float32")
    D, I = faiss_index.search(q_arr, k)
    idxs = I[0].tolist()
    return [chunks[i] for i in idxs if 0 <= i < len(chunks)]

# --------------------------- 3.8 Cross-encoder reranker ------------------------

reranker = CrossEncoder(cross_encoder_name)

_ws_collapse = re.compile(r"\s+")

def _norm_ws(s: str) -> str:
    return _ws_collapse.sub(" ", s).strip()

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = _norm_ws(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def rerank_chunks(question: str, candidate_chunks: List[str], m: int = top_m) -> List[str]:
    """
    Score (question, chunk) pairs with the cross-encoder and keep top-m.
    """
    if not candidate_chunks:
        return []

    pairs = [(question, _norm_ws(c)) for c in candidate_chunks]
    scores = reranker.predict(pairs)  # higher is better
    order = np.argsort(-np.asarray(scores))  # descending
    selected = [candidate_chunks[i] for i in order[:m]]
    return dedupe_preserve_order(selected)

# --------------------------- 3.9 Q&A with ChatGPT ------------------------------

SYS_PROMPT = (
    "You are a knowledgeable assistant that answers questions based on the provided context. "
    "If the answer is not in the context, say you don’t know."
)

def _chat_completion(system_prompt: str, user_prompt: str) -> str:
    """
    Call Chat Completions API. Try spec-compliant parameters first; fall back if the model rejects them.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    attempts: List[Tuple[str, dict]] = [
        # SPEC: temperature=0.0 and max_tokens=500
        ("gpt-5",      {"temperature": 0.0, "max_tokens": 500}),
        # Tolerant fallbacks for various model/version quirks
        ("gpt-5-mini", {"max_completion_tokens": 500}),
        ("gpt-5",      {"max_completion_tokens": 500}),
        ("gpt-5-mini", {"max_tokens": 500}),
    ]

    for model, kwargs in attempts:
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
        except Exception:
            continue
    return ""  # fall back to extractive

_sent_split = re.compile(r"(?<=[\.\!\?])\s+")
_token_split = re.compile(r"\w+")

def _extractive_answer(question: str, context: str) -> str:
    """
    Super-simple extractive fallback: pick the sentence(s) in context with the highest token overlap.
    Keeps the app useful if the API is unavailable or rejects parameters.
    """
    q_tokens = set(map(str.lower, _token_split.findall(question)))
    sentences = _sent_split.split(context)
    scored = []
    for s in sentences:
        s_tokens = set(map(str.lower, _token_split.findall(s)))
        score = len(q_tokens & s_tokens)
        scored.append((score, s.strip()))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for sc, s in scored[:2] if s]
    return top[0] if top else ""

def answer_question(question: str) -> str:
    """
    Retrieve -> rerank -> synthesize with ChatGPT (or extract if API fails).
    """
    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)

    context = "\n\n".join(_norm_ws(c) for c in relevant_chunks) if relevant_chunks else ""
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    if openai.api_key:
        llm = _chat_completion(SYS_PROMPT, user_prompt)
        if llm:
            return llm

    # Extractive fallback (no API or API error)
    return _extractive_answer(question, context) or "I don’t know."

# --------------------------- 3.10 Interactive loop -----------------------------

if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("exit", "quit"):
            break

        ans = answer_question(question)
        print("Answer:", ans)
