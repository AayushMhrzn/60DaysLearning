# Day 54: Exploring Vector Databases with Pinecone

dived into **vector databases** — focusing on **Pinecone** — to store and search vector embeddings generated from text documents. This exploration helps us understand how modern semantic search works beyond traditional keyword matching.

## Semantic Search
Semantic Search is a search technique that understands the meaning or context of a query — rather than just matching keywords. It enables machines to find results that are conceptually similar to a user's input, even if they don't share the exact words.

| Feature       | Traditional Search            | Semantic Search                                           |
| ------------- | ----------------------------- | --------------------------------------------------------- |
| Based on      | Keywords                      | Meaning                                                   |
| Matching      | Exact word match              | Contextual similarity                                     |
| Tech          | Inverted index (TF-IDF, BM25) | Embeddings + vector similarity                            |
| Example Query | `"apple"`                     | `"fruit good for health"` returns "apple", "banana", etc. |

---

## What We Did

1. **Extracted Text from PDFs** using PyMuPDF (fitz)  
2. **Generated Embeddings** using Google Gemini embedding model  
3. **Uploaded vectors + metadata to Pinecone** (called upsert)  
4. **Explored how Pinecone stores and queries vectors**  
5. **Understood similarity scores and their meaning**

---

## Key Concepts & Theory

### 1. Text Embeddings

- **What are embeddings?**  
  Numeric vector representations of text that capture its semantic meaning.  
- **Example:**  
  `"apple"` and `"fruit"` have embeddings close in vector space because they’re related in meaning.  
- **Dimension:**  
  Typically fixed length (e.g., 768) — each number is a coordinate in a high-dimensional space.

---

### 2. Vector Database — What & Why?

- **Definition:**  
  A vector database is a specialized database designed to store, index, and query high-dimensional vectors efficiently.

- **Why normal databases can’t do this well:**  
  Traditional databases are optimized for exact matches or simple numeric/string queries, but **vector similarity search** requires fast approximate nearest neighbor (ANN) search in large, high-dimensional spaces.

- **How vector databases work:**  
  They use algorithms like **HNSW (Hierarchical Navigable Small World)** graphs, **IVF (Inverted File Index)**, or **PQ (Product Quantization)** to index vectors for fast similarity searches.

- **Example use case:**  
  Suppose you have 1 million document embeddings (vectors). You want to find the top 5 most semantically similar documents to a query text. The vector DB can return these results **in milliseconds**, which would be very slow using brute-force search.

---

### 3. Upsert (Upload + Insert)

- Adding vectors + metadata into Pinecone.  
- Each vector gets a unique **ID** and associated metadata (like the original text).  
- Example:  
  ```python
  vector_id = "doc-0"  
  vector = [0.12, 0.45, ..., 0.89]  # 768-dimensional vector  
  metadata = {"text": "This is the document content."}  
  pinecone_index.upsert([(vector_id, vector, metadata)])
  ```

---

### 4. Query Vectors & Similarity Search

- When a **query** is given (e.g., user question), it is also converted into an embedding (query vector).  
- Pinecone compares the query vector with stored vectors and returns the most **similar** ones.  
- **Similarity metric:** Cosine similarity is common.  
- **Cosine similarity formula:**  
  ```
  cosine_similarity = (A · B) / (||A|| * ||B||)
  ```  
  where `·` is dot product and `|| ||` is vector magnitude.

---

### 5. Similarity Score Explained

- Score ranges from -1 to 1 (usually 0 to 1 after normalization).  
- Score close to 1 means vectors are very similar.  
- Score near 0 means little or no similarity.  
- Pinecone UI shows these scores even when viewing a vector because it auto-runs a similarity query behind the scenes (e.g., a vector compared with itself has score ~1).

---

## Example Workflow Recap

1. **Extract text** from `document1.pdf` → `"My introduction is Aayush Maharjan..."`  
2. **Embed text** using Gemini → `[0.12, 0.45, ..., 0.89]` (vector)  
3. **Upsert** into Pinecone with ID `doc-0` and metadata `{"text": "My introduction is Aayush Maharjan..."}`  
4. **Query** Pinecone with `"Who is Aayush Maharjan?"` → embed query text  
5. Pinecone returns top vectors with similarity scores, e.g.:  
   ```
   ID: doc-0, score: 0.9999, text: "My introduction is Aayush Maharjan..."
   ```

---

## Why This Matters

- Vector DBs enable **semantic search** beyond keyword matching.  
- They allow applications like chatbots, recommendation systems, and knowledge bases to find **meaningfully related** information quickly.  
- Understanding embeddings + vector DBs is key to building **next-gen AI-powered apps**.

---

# Summary

| Term               | Definition                                   |
|--------------------|----------------------------------------------|
| Embeddings         | Numeric vectors representing text meaning    |
| Vector Database    | DB optimized for storing and searching vectors|
| Upsert             | Upload vectors + metadata to DB                |
| Query Vector       | Embedding of user input used for searching     |
| Similarity Score   | Measure of closeness between vectors (0-1)     |
