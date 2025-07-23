#  Day 53 - RAG-Powered Chatbot with File Upload and QA

Today, upgraded the chatbot app to support **Retrieval-Augmented Generation (RAG)** using a PDF document as context. This allows users to upload a document, and the chatbot can intelligently answer questions based on the content of that file. Below is a detailed breakdown of what was implemented, how it works, and the core components involved.

---

##  Updates Implemented

### 1. **File Upload & Processing**

* We added a sidebar file uploader that accepts `.pdf` documents.
* Once uploaded, the file is processed using `PyMuPDFLoader` to extract textual content.
* Text is split into manageable chunks using `RecursiveCharacterTextSplitter` (chunk size: 800, overlap: 150).

### 2. **Embedding Creation**

* We use `HuggingFaceEmbeddings` with the model `sentence-transformers/all-MiniLM-L6-v2` to convert each chunk of text into **vector embeddings**.
* These embeddings are numerical representations that capture semantic meaning of text.

### 3. **FAISS Vector Store**

* We store the embeddings using **FAISS (Facebook AI Similarity Search)**.
* FAISS enables **fast vector similarity search**, so we can quickly retrieve the most relevant chunks of text from the document when a user asks a question.

### 4. **Conversational Retrieval Chain**

* We combined the power of an LLM (Groq’s `mixtral-8x7b-32768`) with a **retriever** that queries the vector database.
* We used `ConversationalRetrievalChain` from LangChain to:

  * Accept a user query
  * Retrieve top relevant context chunks from the document using FAISS
  * Pass both the query and context into the LLM to generate a response

### 5. **Seamless UI Integration**

* If a document is uploaded, the app switches to RAG-mode.
* If no document is uploaded, the app functions as a general chatbot with memory and selectable modes like "Python Helper" or "Nepali Translator".

---

##  Core Concepts Explained

###  What is Retrieval-Augmented Generation (RAG)?

RAG enhances language models by allowing them to retrieve relevant context from external documents. Instead of relying only on the model’s internal knowledge, RAG uses a retriever + generator architecture:

* **Retriever**: Finds relevant context from the uploaded file (via vector similarity).
* **Generator (LLM)**: Uses that context + the query to generate accurate and grounded answers.

###  What is an Embedding?

An **embedding** is a dense numerical vector that represents the meaning of text in a high-dimensional space. Similar texts have similar embeddings. These are generated using models like `all-MiniLM-L6-v2` from the `sentence-transformers` library.

###  What is Sentence Transformer?

`sentence-transformers` is a library that provides pre-trained transformer models optimized for producing **sentence-level embeddings**. Unlike vanilla transformers, these models are tuned for tasks like semantic similarity and clustering.

###  What is Vector Similarity Search?

This is the process of finding the most similar vector(s) to a query vector. In our app:

* User query is converted to a vector.
* FAISS finds the document chunks with the closest vectors (using distance metrics like cosine similarity).

###  What is FAISS?

**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search and clustering of dense vectors. It allows us to:

* Index thousands of embeddings
* Search and retrieve top-k similar vectors
* Do it all **very fast**, even for large datasets

---


###  Interact via UI

* Choose a mode from the sidebar (General Assistant, Python Helper, etc.)
* Upload a PDF to enable RAG mode
* Type your query in the chat input box
* Responses will reflect either general LLM chat or PDF-based QA depending on context

---

##  Example Use Case

Upload a PDF containing your syllabus. Ask:

> "What are the main topics in Unit 4?"

The chatbot retrieves the relevant chunk from the PDF, feeds it into the LLM, and provides an accurate answer.

---

