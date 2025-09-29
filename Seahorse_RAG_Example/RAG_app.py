import logging
from transformers import logging as transformers_logging
import warnings
from dotenv import load_dotenv  # Make sure this is imported
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder  # <-- ADDED
import numpy as np
import faiss

# Load environment variables from .env file
load_dotenv()

# Set log levels
transformers_logging.get_logger("langchain.text_splitter").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Retrieve OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure your .env file has OPENAI_API_KEY set.")

openai.api_key = api_key

# Read contents of Selected_Document.txt into text variable
with open("Selected_Document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# -----------------------------
# Parameters
# -----------------------------
chunk_size = 500
chunk_overlap = 100
model_name = "sentence-transformers/all-distilroberta-v1"

# Retrieve K with FAISS, then re-rank to M with a cross-encoder
top_k = 20                             # <-- CHANGED (from 5)
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # <-- ADDED
top_m = 8                              # <-- ADDED

# Split text into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = text_splitter.split_text(text)

# Load model and encode chunks (bi-encoder)
embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings = np.array(embeddings).astype('float32')

# Initialize FAISS index and add embeddings
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# -----------------------------
# Retrieval (bi-encoder + FAISS)
# -----------------------------
def retrieve_chunks(question: str, k: int = top_k):
    """
    Encode the question and search the FAISS index for top k similar chunks.

    Args:
        question (str): The input question string.
        k (int): Number of nearest chunks to retrieve (default: top_k).

    Returns:
        List[str]: List of candidate text chunks.
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec).astype('float32')
    distances, I = faiss_index.search(q_arr, k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# Re-ranking (cross-encoder)
# -----------------------------
# Initialize the cross-encoder once
reranker = CrossEncoder(cross_encoder_name)

def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for it in items:
        key = " ".join(it.split())  # normalize whitespace
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    """
    Score (question, chunk) pairs with a cross-encoder and return the top-m chunks.
    """
    if not candidate_chunks:
        return []
    pairs = [(question, c) for c in candidate_chunks]
    scores = reranker.predict(pairs)  # higher = more relevant
    ranked = sorted(zip(candidate_chunks, scores), key=lambda x: float(x[1]), reverse=True)
    best = [c for c, _ in ranked[:m]]
    return _dedupe_preserve_order(best)

# -----------------------------
# QA with LLM
# -----------------------------
def answer_question(question: str) -> str:
    """
    Retrieves candidate chunks, re-ranks them, and uses OpenAI's Chat Completions API to answer.
    """
    # Retrieve candidate chunks via FAISS
    candidates = retrieve_chunks(question)

    # Re-rank to final context
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)

    # Combine chunks into a single context string separated by double newlines
    context = "\n\n".join(relevant_chunks)

    # System prompt defining the assistant's behavior
    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don’t know."
    )

    # User prompt including the context and the question
    user_prompt = f"""Context:
{context}

Question: {question}

Answer:
"""

    # Call OpenAI Chat Completions with the prompts and parameters
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # You can switch to "gpt-4o" or a newer model if available
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    # Return the assistant's reply text, stripped of whitespace
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
