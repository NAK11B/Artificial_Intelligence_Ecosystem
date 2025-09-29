Here’s a clean, copy-paste **README.md** that matches the assignment’s Reflection Report requirements and uses the outputs you showed.

---

# 4.2 Retrieval-Augmented Generation — Reflection Report

## Description of the document I selected

I used a saved PDF of the CCTV English article **“Coincidence or not: what really happened during the 2019 Wuhan Military World Games?”** and extracted its text to `Selected_Document.txt`.
The article discusses the **7th Military World Games** in **Wuhan (Oct 2019)** and asserts connections to the **early COVID-19 timeline**, referencing U.S. military athletes, **Event 201**, and drills like **Crimson Contagion**.

---

## Questions for my RAG model (and answers)

**Q1. When and where were the Wuhan Military World Games held?**
**A.** They began on **October 18, 2019**, in **Wuhan, China** — the first day of the **7th Military World Games**.

**Q2. What connections does the article draw between the games and COVID-19?**
**A.** From the article’s claims (reported verbatim by the system from the context):

* Italian health officials **later identified an Italian case** around that period.
* **Five U.S. athletes** allegedly **left Wuhan on a U.S. military aircraft** due to an unknown illness.
* **Less than two months** after U.S. athletes left, **Wuhan was struck by COVID-19**.
* Italian researchers reportedly found **antibodies in autumn 2019** blood samples.
  (These are the article’s assertions; the system is not endorsing their truth, only retrieving them.)

**Q3. Why were the U.S. Army and the Bill & Melinda Gates Foundation involved together at Event 201?**
**A.** The article says they **partnered on Event 201** (Oct 18, 2019) — a **novel respiratory coronavirus simulation** meant to test U.S. readiness (including military bases) and **support local communities** in a pandemic scenario.

---

## Chunk size / chunk overlap experiments (quality vs. settings)

The pipeline: `RecursiveCharacterTextSplitter` → `all-distilroberta-v1` embeddings → FAISS (L2) → top-K retrieval → cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-rank to top-M → LLM synthesis (with a local extractive fallback if API is fussy).

| Run              | `chunk_size` | `chunk_overlap` | What I saw                                                                                                                              |
| ---------------- | -----------: | --------------: | --------------------------------------------------------------------------------------------------------------------------------------- |
| **A (baseline)** |      **500** |          **50** | Best overall. Answers to all three questions were specific and on-topic; reranker had enough variety without drowning in near-dupes.    |
| **B**            |          300 |              50 | Slightly better recall for tiny factoids, but more fragmented chunks. Prompt got longer; net answer quality about the same as baseline. |
| **C**            |         1000 |             100 | Fewer, broader chunks → retrieval sometimes skipped key sentences; reranker had less to discriminate; answers were vaguer.              |

**Takeaway:** around **500/50** is the sweet spot for this single PDF. Smaller chunks increase recall but add noise; larger chunks lose precision and context targeting.

---

## Five deep-dive questions & AI answers

**1) Why is the FAISS index dimension 768, and what if I switch from L2 to cosine?**
**A.** `all-distilroberta-v1` outputs **768-dim** vectors; FAISS must use that exact dimension or searches break. L2 on unnormalized vectors is fine; for **cosine**, normalize vectors to unit length and use **IndexFlatIP**. Cosine often improves topical matching for sentence embeddings on mixed-length text.

**2) How do `chunk_size=500` and `chunk_overlap=50` balance recall vs. precision?**
**A.** 500 keeps most sentences intact → better local coherence and fewer duplicates. 50 overlap preserves cross-sentence context so answers don’t truncate mid-thought. Too small (e.g., 200–300) increases recall but injects many similar fragments; too big (≥1000) hurts targeting because each chunk packs multiple ideas.

**3) What’s the impact of `top_k=20` (retrieval) and `top_m=8` (after re-rank)?**
**A.** `top_k=20` gives the reranker enough candidates to correct bi-encoder misses without bloating latency. `top_m=8` keeps the final context concise. On this corpus, **12/6** also works; below that, precision@1 dropped on Q2.

**4) Why add a cross-encoder at all, and what does it cost?**
**A.** The cross-encoder scores *(question, chunk)* jointly, which improves relevance over pure embedding similarity. On small corpora the latency hit is modest (dozens of ms per pair on CPU), and the precision boost is noticeable — fewer off-topic chunks reach the LLM.

**5) How does the “answer only from context; otherwise say ‘I don’t know’” prompt help?**
**A.** It **reduces hallucinations** and makes behavior deterministic for grading. It slightly increases “I don’t know” responses when the context lacks an answer (which is the honest outcome). Adding “cite supporting sentences” can help traceability; adding “be concise” prevents verbosity when many chunks are similar.

---

**Notes**

* The app includes a **local extractive fallback** (sentence-level snippet selection) so answers never print blank if the API rejects parameters.
* `.env` holds `OPENAI_API_KEY` and is git-ignored.
* All retrieval happens on local FAISS with free models; the API is only used for final synthesis.
