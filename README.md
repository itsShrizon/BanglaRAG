# GraphRAG API: Aparichita Story Query Service

An API for querying the story "Aparichita" using a Retrieval-Augmented Generation (RAG) pipeline. It retrieves context from multiple sources and generates answers using a language model.

---

## Setup Guide
1. **Clone the repository**
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the API server**:
   ```powershell
   python src/api.py
   ```
   The server will start at `http://0.0.0.0:8000`.

---

## Used Tools, Libraries, and Packages
- **FastAPI**: Web API framework
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation
- **LangChain**: RAG pipeline and LLM integration
- **OpenAI**: Embedding model for semantic search
- **Google Gemini**: Generative AI for text extraction and answer generation
- **Pinecone**: Vector database for fast semantic search
- **AstraDB**: Multi-modal vector database for hybrid retrieval
- **PyMuPDF (fitz)**: PDF to image conversion for robust text extraction
- **Other**: Standard Python libraries

See `requirements.txt` for the full list.

---

## API Documentation

### POST `/query`
Query the RAG pipeline with a question.

**Request Body:**
```json
{
  "query": "<your question>"
}
```

**Response:**
```json
{
  "query": "<your question>",
  "answer": "<generated answer>",
  "sources": ["<context chunk 1>", "<context chunk 2>", ...]
}
```

#### Sample Queries and Outputs

**English Example**
Request:
```json
{
  "query": "Who is the main character?"
}
```
Response:
```json
{
  "query": "Who is the main character?",
  "answer": "The main character is Aparichita.",
  "sources": [
    "Chunk 1 text...",
    "Chunk 2 text..."
  ]
}
```

**Bangla Example**
Request:
```json
{
  "query": "অপরিচিতা গল্পের প্রধান চরিত্র কে?"
}
```
Response:
```json
{
  "query": "অপরিচিতা গল্পের প্রধান চরিত্র কে?",
  "answer": "প্রধান চরিত্র অপরিচিতা।",
  "sources": [
    "চাঙ্ক ১ টেক্সট...",
    "চাঙ্ক ২ টেক্সট..."
  ]
}
```

---

## Text Extraction Strategy

Text extraction was performed using the **fitz (PyMuPDF)** library to convert PDF pages into images (see `extract.py`). Then, **Google Gemini's Generative AI API** (`google.generativeai`) was used to extract text from those images. This approach was chosen because Bengali PDFs often have formatting issues, making direct text extraction unreliable. By converting pages to images and using an AI model for OCR and text extraction, the process is more robust against inconsistent fonts, layouts, and embedded graphics.

**Formatting challenges included:**
- Page headers/footers, titles, and non-sentence content (like "HSC 26" or "10 MINUTE SCHOOL") that needed to be filtered out.
- Ensuring sentence boundaries and clean extraction, which required both AI-based and regex-based fallback methods.
- Handling Bengali script and maintaining sentence integrity.

These issues were addressed by using AI for extraction and additional cleaning steps in the processing pipeline.

---

## Chunking and Retrieval Strategy

The project uses a **sentence-based chunking strategy** (see [`src/chunk.py`](src/chunk.py)). Bengali text is split into individual sentences using Google Gemini and regex fallback, then grouped into clusters (typically 7 sentences per chunk) using KMeans clustering based on semantic embeddings.

**Why this works well for semantic retrieval:**
- **Semantic coherence:** Related sentences are grouped together, preserving context and meaning.
- **Avoids splitting important information:** Sentence boundaries prevent breaking up key details.
- **Improved relevance:** Clustering by semantic similarity ensures each chunk is topically focused, making retrieval more accurate.
- **Hybrid retrieval:** Chunks are indexed in both Pinecone and AstraDB, and results are combined for higher recall and robustness.

This approach maximizes the likelihood that retrieved chunks contain contextually relevant information for user queries.

---

## RAG Pipeline and Hybrid Retrieval

This script creates an advanced conversational AI system (a RAG pipeline) that answers questions about a story.

**How it works:**
- **Initialization:** Loads API keys from your `.env` file and connects to four different services:
  - Google Gemini: To generate the final answers.
  - OpenAI: To create the numerical embeddings (vector representations) of the text.
  - Pinecone: A vector database for searching text.
  - AstraDB: Another vector database for searching text.
- **Short-Term Memory:** Maintains a chat history. When you ask a follow-up question (e.g., "What was his age?"), it first uses Gemini to rephrase it into a complete, standalone question (e.g., "What was Shambhunath Babu's age?"). This makes the search much more accurate.
- **Hybrid Retrieval:** Takes the standalone question and searches for relevant context in both Pinecone and AstraDB simultaneously. Combines the results and removes any duplicate information to create a rich, comprehensive context.
- **Answer Generation:** Sends the retrieved context, the original question, and the chat history to the powerful Gemini 1.5 Pro model. It instructs the model to act like a literary analyst and answer the question based only on the provided information.

In short, it's a sophisticated chatbot that remembers your conversation and pulls information from two separate databases to provide the most accurate possible answer.

---

## Embedding Model and Similarity Search

- **Embedding Model:** Uses OpenAI's `text-embedding-3-large` model to generate semantic embeddings for each chunk (see [`src/pinecone_embed.py`](src/pinecone_embed.py)).
- **Why this model?** It provides high-quality, language-agnostic vector representations that capture the underlying meaning and context of text, even for Bengali content.
- **How does it capture meaning?** The model transforms text into high-dimensional vectors where semantically similar texts are close together in vector space, allowing the retrieval system to match user queries with relevant story chunks based on meaning, not just keywords.

**Similarity Method:**  
- Uses **Cosine Similarity** for comparing vectors. This measures the angle between two vectors, focusing on semantic similarity rather than just word overlap.

**Why this setup?**
- **Cosine Similarity:** Industry standard for text-based RAG; excels at capturing semantic meaning.
- **Hybrid Storage (Pinecone + AstraDB):** Combines fast, dense vector search (Pinecone) with rich, structured metadata and future graph queries (AstraDB), increasing recall and robustness.

---

## Ensuring Meaningful Comparison and Handling Vague Queries

**Meaningful Comparison:**
- Uses semantic search via embeddings, not just keyword matching.
- Deep meaning via OpenAI embeddings: Both queries and document chunks are converted to vectors that capture intent and context.
- Cosine similarity retrieves conceptually related chunks, even if they don't share exact words.

**Handling Vague or Context-Dependent Queries:**
- Uses chat history and Gemini to rewrite vague queries into standalone, context-rich questions before searching.
- Ensures accurate retrieval even for conversational or ambiguous questions.

---

## Evaluation Matrix (if implemented)

If evaluation is implemented, results and metrics (e.g., accuracy, F1-score, context relevance) will be found in `data/evaluation/` or described here.

---

## Relevance of Results and Improvements

**Do the results seem relevant?**
- **Partially.** The retrieval step works well; the system often finds the correct sentences. However, the final generation step may produce incorrect answers, indicating the issue is with answer synthesis, not retrieval.

**What might improve them?**
- **Better Chunking:** Although the current sentence-based chunking strategy is well-suited for Bangla literature and tasks like this, due to limited time and job pressure, I was not able to implement even higher-quality chunking methods (such as paragraph-based or context-preserving chunking). With more time, further improvements in chunking could dramatically enhance answer accuracy and overall system performance.


