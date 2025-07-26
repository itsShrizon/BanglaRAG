"""
Industry-grade script to load chunked documents from a JSON file, generate
embeddings using OpenAI, and upsert them into a specified AstraDB collection.

---
Features:
- Loads credentials securely from a .env file.
- Uses command-line arguments for easy configuration.
- Implements structured logging for clear, informative output.
- Processes and uploads documents in batches with a real-time progress bar.
- Includes robust error handling for file and database operations.

---
Prerequisites:
1. A `.env` file in the project's root directory containing:
   - ASTRA_DB_API_ENDPOINT="your_astra_db_endpoint"
   - ASTRA_DB_APPLICATION_TOKEN="your_astra_db_token"
   - OPENAI_API_KEY="your_openai_api_key"

2. Required Python packages installed:
   pip install python-dotenv langchain-openai langchain-astradb tqdm

---
How to Run:
Open your terminal and run the script directly. You can optionally override
the collection name.

Example:
  python src/astraldb_embed.py
  python src/astraldb_embed.py --collection "another_collection"
"""

# --- Dependencies ---
import os
import json
import logging
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

# --- 1. Static Configuration ---
# Hardcoded path to the input file.
INPUT_FILE = Path("data/processed/merged_chunks_metadata.json")


# --- 2. Logging Configuration ---
# Set up a logger to provide clear, timestamped feedback during execution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- 3. Script Configuration & Argument Parsing ---
def get_script_config() -> argparse.Namespace:
    """
    Parses command-line arguments to configure the script's behavior.
    This makes the script flexible without needing to change the code.
    """
    parser = argparse.ArgumentParser(
        description="Embed and upload documents to AstraDB.",
        formatter_class=argparse.RawTextHelpFormatter # Improves help text formatting
    )

    # Optional arguments: these have default values but can be overridden.
    parser.add_argument(
        "--collection",
        type=str,
        default="combined_rag_db",
        help="Name of the target AstraDB collection (default: %(default)s).",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="text-embedding-3-large",
        help="Name of the OpenAI embedding model to use (default: %(default)s).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,  # AstraDB's recommended max batch size
        help="Number of documents to upsert in each batch (default: %(default)s).",
    )
    return parser.parse_args()


# --- 4. Data Loading ---
def load_documents_from_json(file_path: Path) -> List[Document]:
    """
    Loads, validates, and parses documents from a specified JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of LangChain Document objects, ready for embedding.
    """
    # Defensive check: ensure the file exists before trying to read it.
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    logger.info(f"Loading documents from {file_path}...")
    try:
        # Read the file content
        data = json.loads(file_path.read_text(encoding="utf-8"))
        chunks = data.get("chunks")
        metadata = data.get("metadata")

        # Validate the structure of the JSON file.
        if not isinstance(chunks, list) or not isinstance(metadata, list) or len(chunks) != len(metadata):
            raise ValueError("JSON must contain 'chunks' and 'metadata' lists of the same length.")

        # Combine chunks and metadata into LangChain's standard Document format.
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(chunks, metadata)
        ]
        logger.info(f"Successfully loaded {len(docs)} documents.")
        return docs
    except json.JSONDecodeError:
        logger.error(f"Error: Failed to decode JSON from {file_path}. Please check file for syntax errors.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading documents: {e}")
        raise


# --- 5. Main Execution ---
def main():
    """Main function to orchestrate the embedding and upload process."""
    # Load environment variables (API keys, etc.) from the .env file.
    load_dotenv()
    # Get all configuration settings from the command line.
    config = get_script_config()

    try:
        # Step 1: Load the documents from the hardcoded file path.
        documents = load_documents_from_json(INPUT_FILE)
        if not documents:
            logger.warning("No documents found to process. Exiting.")
            return

        # Step 2: Initialize the embedding model from OpenAI.
        logger.info(f"Initializing OpenAI embedding model: {config.embed_model}")
        embeddings = OpenAIEmbeddings(model=config.embed_model)

        # Step 3: Connect to the AstraDB vector store.
        # This will create the collection if it doesn't exist.
        logger.info(f"Connecting to AstraDB collection: '{config.collection}'")
        # FIX: Removed the 'dimension' argument as it's no longer needed.
        # The library now infers the dimension from the embedding model automatically.
        vstore = AstraDBVectorStore(
            embedding=embeddings,
            collection_name=config.collection,
        )

        # Step 4: Upload documents in batches to avoid overwhelming the API.
        # The 'tqdm' library creates a helpful progress bar in the terminal.
        logger.info(f"Upserting {len(documents)} documents in batches of {config.batch_size}...")
        for i in tqdm(range(0, len(documents), config.batch_size), desc="Upserting to AstraDB"):
            batch = documents[i : i + config.batch_size]
            vstore.add_documents(batch)

        logger.info(f"✅ Successfully inserted {len(documents)} documents into '{config.collection}'.")

    # Gracefully handle specific, expected errors.
    except FileNotFoundError as e:
        logger.error(f"Aborting process. {e}")
    except ValueError as e:
        logger.error(f"Aborting due to data or configuration error: {e}")
    # Catch any other unexpected errors.
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


# This standard Python construct ensures the main() function runs only
# when the script is executed directly.
if __name__ == "__main__":
    main()
