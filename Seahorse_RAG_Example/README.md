## Description of the document you selected

I selected the Wikipedia article about seahorses.


---

## 5 important questions and answers

### 1. Why do we need to chunk the text before storing it in the vector database?

**Answer:**
Chunking breaks long documents into smaller, manageable pieces that fit within the token limits of language models. It also helps preserve local context, making retrieval more precise. If you indexed entire documents as single chunks, the embeddings would be less focused, and retrieval would be less accurate. Overlapping chunks help ensure continuity across boundaries.

---

### 2. How is the retrieved text processed before being stored in the vector database?

**Answer:**
After the agent collects the raw text (from a web page or PDF), it is split into smaller chunks using tokenization tools like `tiktoken`. These chunks are typically overlapping segments of a fixed token size (e.g., 300 tokens with 30 token overlap) to preserve context. Each chunk is then converted into a high-dimensional vector embedding using an embedding model like OpenAI’s `text-embedding-3-small`. These embeddings are indexed in a vector store (e.g., FAISS) for efficient similarity search.

---

### 3. How does the system find relevant pieces of information to answer a user’s question?

**Answer:**
When a user asks a question, the system embeds the question into the same vector space as the stored chunks. It then queries the vector store to find the top-k chunks most similar to the question vector. These relevant chunks provide the contextual information the model needs to generate accurate and informed answers.

---

### 4. How does the language model use the retrieved chunks to generate an answer?

**Answer:**
The retrieved chunks are combined into a context block that is provided as part of the prompt to the language model (e.g., GPT-4). The prompt instructs the model to answer the user’s question using **only** the provided context. This constrains the model’s generation to grounded information, improving accuracy and reducing hallucination compared to answering without external knowledge.

---

### 5. What types of embeddings are used, and why are they important for this system?

**Answer:**
The system uses dense vector embeddings produced by transformer-based embedding models (like OpenAI’s `text-embedding-3-small`). These embeddings capture the semantic meaning of text chunks and questions in a numerical format, enabling the vector store to perform similarity searches based on meaning, not just keywords. This semantic search is essential to finding the most relevant context for the user’s queries and is the foundation of how RAG improves the language model’s answers.

---

## 3 Questions and Answer Quality

**Your question:** What is a group of seahorses called?
**Answer:** A group of seahorses is not specifically mentioned in the provided context.


**Your question:** Does a seahorse have gills?
**Answer:** Yes, a seahorse has gills.


**Your question:** How many species of seahorses are there?
**Answer:** There are several species of seahorses mentioned in the provided context, including H. kellogii, H. histrix, H. kuda, H. trimaculatus, H. mohnikei, Hippocampus erectus, and H. zosterae. However, the exact number of seahorse species is not explicitly stated in the context.

The quality of the answers wasn't great because the Wikipedia article wasn't as detailed as some of them are; however, the model was good at saying when the context didn't supply the answer rather than halucinating.

To test what happens when I change the chunk size and overlap size, I asked the RAG model to "Tell me about seahorse courtship", which is something I know the Wikipedia article had a section on.

**Chunk size = 1000**
**Overlap = 500**

**Answer:** Seahorses exhibit four phases of courtship that involve clear behavioral changes and changes in the intensity of the courtship act. Phase 1 is the initial courtship phase that occurs one or two days before physical copulation, where potential mates brighten in color, quiver, and display rapid side-to-side body vibrations. Phase 2 involves the female pointing behavior, followed by the male responding with the same behavior. In Phase 3, both males and females brighten and assume the pointing position, with the male eventually departing. The final phase includes multiple bouts of courtship where the male and female rise together in a water column, culminating in mid-water copulation where the female transfers her eggs into the male's brood pouch. Courtship behavior in seahorses can last for several days and involves synchronized movements and reproductive states to ensure successful egg transfer.

**Chunk size = 500**
**Chunk overlap = 100**

**Answer:** Seahorse courtship involves four phases that include behavioral changes and changes in the intensity of courtship acts. The courtship behavior includes brightening in color, quivering, rapid side-to-side body vibrations, swimming side by side holding tails, gripping the same strand of sea grass with their tails, and engaging in synchronized movements like the "predawn dance" and "true courtship dance." The final phase of courtship includes bouts of courtship where the male and female rise together in the water column, with the female eventually transferring her eggs to the male. Male seahorses are more aggressive during courtship, engaging in behaviors like tail-wrestling and head snapping to compete for female attention.


**Chunk size = 100**
**Chunk overlap = 50**

**Answer:** Seahorse courtship involves four phases with clear behavioral changes, where displays are performed alternately by both the male and the female seahorse. Before breeding, seahorses may court for several days. The seahorses display no courtship behavior, and females are not bright; males will usually display.


In general, it seems like the shorter the chunk size and the less the chunk overlap, the less detailed the answer is.