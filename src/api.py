import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from combined_rag import retriever_chain, format_and_combine_context, final_prompt, llm

# --- FastAPI App ---
app = FastAPI(
    title="GraphRAG API",
    description="An API to query the 'Aparichita' story."
)

class QueryRequest(BaseModel):
    query: str

# --- API Endpoint ---
@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    This endpoint takes a user's question, retrieves context using the retriever_chain,
    generates a final answer using the RAG pipeline, and returns both.
    """
    # 1. Retrieve context from both sources
    retrieved_docs = retriever_chain.invoke(request.query)
    combined_context = format_and_combine_context(retrieved_docs)

    # 2. Generate the final answer using the prompt and LLM
    generation_chain = final_prompt | llm
    answer_obj = generation_chain.invoke({"context": combined_context, "question": request.query})
    answer = answer_obj.content if hasattr(answer_obj, "content") else str(answer_obj)

    # 3. Format the response, extracting the text chunks
    sources = combined_context.split('\n\n')

    return {
        "query": request.query,
        "answer": answer,
        "sources": sources
    }

# --- API Documentation ---

## Overview

This API provides a single endpoint to query the "Aparichita" story using a Retrieval-Augmented Generation (RAG) pipeline. It retrieves relevant context from two vector databases (Pinecone and AstraDB) and generates an answer using Google's Gemini model.

---

## Endpoint

### `POST /query`

Query the RAG pipeline with a question about the story.

#### Request Body

```json
{
  "query": "string"
}
```

- **query**: *(string, required)*  
  The question you want to ask about the story.

#### Response

```json
{
  "query": "string",
  "answer": "string",
  "sources": [
    "string",
    "string",
    ...
  ]
}
```

- **query**: The original question.
- **answer**: The generated answer in Bengali, based only on the retrieved context.
- **sources**: List of text chunks (context) retrieved from the databases.

#### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
```

#### Example Response

```json
{
  "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
  "answer": "কল্যাণীর বয়স ছিল ১৮ বছর।",
  "sources": [
    "কল্যাণীর বয়স ছিল ১৮ বছর।",
    "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "... (other relevant chunks) ..."
  ]
}
```

---

## Error Handling

- If the answer cannot be found in the context, the API responds with:  
  `'প্রদত্ত তথ্যে এই প্রশ্নের উত্তর খুঁজে পাওয়া যায়নি।'`
- If there is an internal error, a 500 error is returned.

---

## Swagger & Interactive Docs

- Visit `http://localhost:8000/docs` for interactive Swagger UI.
- Visit `http://localhost:8000/redoc` for ReDoc documentation.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)