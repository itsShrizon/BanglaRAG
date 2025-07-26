import os
import re
import json
import logging
from typing import List, Dict, Tuple

import google.generativeai as genai
from sklearn.cluster import KMeans
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# --- Configuration ---
# Sets up logging to show the script's progress in the terminal.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Loads the API key from a .env file.
load_dotenv()

class Config:
    """A 'control panel' for the script. All settings are here."""
    # API and Model Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SENTENCE_DETECTION_MODEL = "gemini-2.5-flash-preview-05-20"
    METADATA_EXTRACTION_MODEL = "gemini-2.5-flash-preview-05-20"
    EMBEDDING_MODEL = "models/text-embedding-004"

    # File Paths
    OUTPUT_DIR = os.path.join("data", "processed")
    INPUT_FILE = os.path.join(OUTPUT_DIR, "extracted_text.txt")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_chunks_metadata.json")

    # The number of sentences to group into a single chunk for analysis.
    SENTENCES_PER_CHUNK = 7

class TextProcessor:
    """This class contains all the logic for processing the text."""
    def __init__(self, config: Config):
        self.config = config
        if not self.config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Please set it in a .env file.")
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def _clean_text(self, text: str) -> str:
        """Removes extra whitespace and non-standard characters from text."""
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[^\w\s.,!?।\u0980-\u09FF]", "", text)
        return text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _detect_sentences(self, text: str) -> List[str]:
        """Uses Gemini to detect sentence boundaries, with a fallback method."""
        sentences = []
        max_chunk_size = 2000  # Process text in 2000-character chunks to avoid payload limits
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i:i + max_chunk_size]
            try:
                model = genai.GenerativeModel(self.config.SENTENCE_DETECTION_MODEL)
                prompt = f"""Split this Bengali text into individual sentences, excluding headers, footers, or non-sentence content (e.g., titles, page numbers, phrases like 'HSC 26' or '10 MINUTE SCHOOL'). Return only the sentences, each on a new line, with no extra formatting, numbering, or explanations.

Text: {chunk}"""
                response = model.generate_content(prompt)
                if response.text:
                    logging.info(f"Gemini response for chunk {i//max_chunk_size + 1}: {response.text[:50]}...")
                    chunk_sentences = [s.strip() for s in response.text.strip().split('\n') if s.strip()]
                    sentences.extend(chunk_sentences)
                else:
                    logging.warning(f"Empty response from Gemini API for chunk {i//max_chunk_size + 1}. Using regex fallback.")
                    chunk_sentences = [s.strip() for s in re.split(r"[।\n]", chunk) if s.strip()]
                    sentences.extend(chunk_sentences)
            except Exception as e:
                logging.error(f"Error detecting sentences for chunk {i//max_chunk_size + 1}: {e}")
                if "429" in str(e):
                    logging.error("Quota exceeded for gemini-2.5-flash-preview-05-20. Check https://ai.google.dev/gemini-api/docs/rate-limits or wait for daily reset.")
                chunk_sentences = [s.strip() for s in re.split(r"[।\n]", chunk) if s.strip()]
                sentences.extend(chunk_sentences)
        if not sentences:
            logging.warning("No valid sentences found in text.")
        return sentences

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _extract_metadata(self, text: str) -> Dict[str, List[str]]:
        """Extracts characters, themes, and settings from a text chunk."""
        try:
            model = genai.GenerativeModel(self.config.METADATA_EXTRACTION_MODEL)
            prompt = f"""From the following Bengali text, extract the characters, themes, and settings. Respond in Bengali. Return only JSON in this format:
```json
{{
  "characters": [],
  "themes": [],
  "settings": []
}}
```

Text: {text[:1000]}"""
            response = model.generate_content(prompt)
            if response.text:
                # Find the JSON block using regex, even if it's surrounded by text
                match = re.search(r"\{.*?\}", response.text, re.DOTALL)
                if match:
                    json_string = match.group(0)
                    try:
                        # Attempt to parse the found JSON string
                        parsed = json.loads(json_string)
                        if isinstance(parsed, dict) and all(key in parsed for key in ["characters", "themes", "settings"]):
                            return parsed
                        else:
                            logging.warning(f"Invalid metadata format: {json_string[:50]}...")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error: {e}. AI response was: {json_string[:50]}...")
                else:
                    logging.warning(f"No JSON object found in the AI response: {response.text[:50]}...")
            else:
                logging.warning("Empty response from Gemini API for metadata extraction.")
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            if "429" in str(e):
                logging.error("Quota exceeded for gemini-2.5-pro. Check https://ai.google.dev/gemini-api/docs/rate-limits or wait for daily reset.")
        return {"characters": [], "themes": [], "settings": []}

    def _get_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """Generates embeddings for sentences using the embedding model."""
        embeddings = []
        max_chars = 10000  # Safe limit to avoid 36,000-byte payload error
        logging.info(f"Processing {len(sentences)} sentences for embeddings...")
        for i, sentence in enumerate(sentences):
            if len(sentence.encode("utf-8")) > 36000:
                logging.warning(f"Splitting long sentence exceeding 36,000 bytes: {sentence[:50]}...")
                # Split sentence into chunks at word boundaries
                words = sentence.split()
                chunks = []
                current_chunk = []
                for word in words:
                    test_chunk = " ".join(current_chunk + [word])
                    if len(test_chunk.encode("utf-8")) > 36000:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                    else:
                        current_chunk.append(word)
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Get embeddings for each chunk and average them
                chunk_embeddings = []
                for chunk in chunks:
                    try:
                        response = genai.embed_content(
                            model=self.config.EMBEDDING_MODEL,
                            content=chunk,
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        chunk_embeddings.append(response["embedding"])
                    except Exception as e:
                        logging.error(f"Error embedding chunk: {e}")
                        chunk_embeddings.append([0.0] * 768)
                
                # Average the embeddings
                if chunk_embeddings:
                    avg_embedding = [sum(vals) / len(vals) for vals in zip(*chunk_embeddings)]
                    embeddings.append(avg_embedding)
                else:
                    embeddings.append([0.0] * 768)
            else:
                try:
                    response = genai.embed_content(
                        model=self.config.EMBEDDING_MODEL,
                        content=sentence,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    embeddings.append(response["embedding"])
                except Exception as e:
                    logging.error(f"Error embedding sentence {i + 1}: {sentence[:50]}... | Error: {e}")
                    embeddings.append([0.0] * 768)  # Fallback zero vector
            
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{len(sentences)} embeddings...")
        logging.info(f"Completed embedding {len(embeddings)} sentences.")
        return embeddings

    def chunk_text(self, text: str) -> Tuple[List[str], List[Dict[str, List[str]]]]:
        """Processes the text into chunks with metadata."""
        text = self._clean_text(text)
        sentences = self._detect_sentences(text)
        if not sentences:
            logging.error("No valid sentences found. Cannot proceed with chunking.")
            return [], []

        embeddings = self._get_sentence_embeddings(sentences)
        num_clusters = max(1, len(sentences) // self.config.SENTENCES_PER_CHUNK)
        logging.info(f"Creating {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        logging.info("Clustering completed. Creating chunks...")

        chunks = []
        metadata = []
        current_chunk = []
        current_size = 0
        current_label = None
        for sent, label in zip(sentences, labels):
            sent_size = len(sent.split())
            if (current_label is not None and label != current_label) or len(current_chunk) >= self.config.SENTENCES_PER_CHUNK:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    logging.info(f"Extracting metadata for chunk {len(chunks)}...")
                    metadata.append(self._extract_metadata(chunk_text))
                current_chunk = [sent]
                current_size = sent_size
                current_label = label
            else:
                current_chunk.append(sent)
                current_size += sent_size
                current_label = label
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            logging.info(f"Extracting metadata for final chunk {len(chunks)}...")
            metadata.append(self._extract_metadata(chunk_text))

        logging.info(f"Created {len(chunks)} chunks. Saving to {self.config.OUTPUT_FILE}...")
        with open(self.config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks, "metadata": metadata}, f, ensure_ascii=False)
        logging.info("Processing completed successfully!")
        return chunks, metadata

if __name__ == "__main__":
    try:
        processor = TextProcessor(Config())
        with open(processor.config.INPUT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        processor.chunk_text(text)
    except FileNotFoundError:
        logging.error(f"Input file {Config.INPUT_FILE} not found. Run extract.py first.")
    except Exception as e:
        logging.error(f"Error processing text: {e}")