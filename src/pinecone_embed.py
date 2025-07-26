"""
Reads data/processed/extracted_text.txt, splits it into semantic chunks,
embeds them using the large OpenAI model, and uploads them directly to a
Pinecone index.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# --- Load Environment Variables ---
# Make sure you have a .env file with your API keys
load_dotenv()

# --- Configuration ---
# File Paths
RAW_TXT = Path("data/processed/extracted_text.txt")

# Chunking Parameters
CHUNK_SIZE = 1600     # Target size for each chunk in characters
CHUNK_OVERLAP = 160   # Number of characters to overlap between chunks

# Pinecone & OpenAI Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# FIX: Updated to use your existing Pinecone index name.
PINECONE_INDEX_NAME = "combined-rag-index"


def main() -> None:
    """
    Main function to read, chunk, embed, and upload text to Pinecone.
    """
    # --- 1. Validate Inputs ---
    if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY]):
        raise ValueError("Please set PINECONE_API_KEY, PINECONE_INDEX_NAME, and OPENAI_API_KEY in your .env file or script.")
    if not RAW_TXT.exists():
        raise FileNotFoundError(f"Input file not found: {RAW_TXT}. Please ensure it exists.")

    # --- 2. Read and Chunk Text ---
    print(f"Reading text from {RAW_TXT}...")
    raw_text = RAW_TXT.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    print(f"Split text into {len(chunks)} chunks.")

    # --- 3. Initialize Services (OpenAI for embeddings, Pinecone for vector storage) ---
    print("Initializing OpenAI embeddings and Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # --- 4. Check for Pinecone Index ---
    # This logic will now connect to your existing index.
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Please ensure the index exists in your Pinecone project.")
        # The script will stop if the index doesn't exist.
        # You can uncomment the create_index block if you need to create it programmatically.
        # print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        # pc.create_index(
        #     name=PINECONE_INDEX_NAME,
        #     dimension=3072,
        #     metric='cosine',
        #     spec=ServerlessSpec(
        #         cloud='aws',
        #         region='us-east-1'
        #     )
        # )
    else:
        print(f"Found existing Pinecone index '{PINECONE_INDEX_NAME}'.")

    index = pc.Index(PINECONE_INDEX_NAME)

    # --- 5. Embed Chunks and Prepare for Upload ---
    print("Embedding chunks... This may take a moment.")
    embedded_chunks = embeddings.embed_documents(chunks)

    vectors_to_upsert = []
    for i, (chunk_text, vec) in enumerate(zip(chunks, embedded_chunks)):
        vectors_to_upsert.append({
            "id": f"chunk_{i}",
            "values": vec,
            "metadata": {"text": chunk_text}
        })

    # --- 6. Upsert to Pinecone in Batches ---
    print(f"Uploading {len(vectors_to_upsert)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  - Upserted batch {i//batch_size + 1}")

    print(f"✓ Successfully uploaded all chunks to Pinecone.")
    print(f"Index stats: {index.describe_index_stats()}")


if __name__ == "__main__":
    main()
