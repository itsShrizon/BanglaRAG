from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
import google.generativeai as genai
import os
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def pdf_to_images(pdf_path, output_dir, start_page=3, end_page=19):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    if start_page < 1 or end_page > pdf_document.page_count or start_page > end_page:
        raise ValueError(f"Invalid page range: {start_page} to {end_page}. PDF has {pdf_document.page_count} pages.")
    image_paths = []
    # Adjust to 0-based indexing; pages 3 to 19 = indices 2 to 18
    for page_num in range(start_page - 1, end_page):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    pdf_document.close()
    return image_paths

@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Wait 2s, 4s, up to 10s
    retry=retry_if_exception_type(Exception)  # Retry on any exception
)
def extract_text_with_gemini(image_path):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        image = Image.open(image_path)  # Load image with PIL
        response = model.generate_content(["Extract all text from this image.", image])
        return response.text if response.text else ""
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        raise

def extract_from_pdf(pdf_path, output_dir, start_page=3, end_page=19):
    try:
        image_paths = pdf_to_images(pdf_path, output_dir, start_page, end_page)
        texts = []
        # Add progress bar for text extraction
        for img in tqdm(image_paths, desc="Extracting text from images", unit="page"):
            text = extract_text_with_gemini(img)
            texts.append(text)
        os.makedirs("data/processed", exist_ok=True)
        output_file = os.path.join("data", "processed", "extracted_text.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        return texts
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []

if __name__ == "__main__":
    pdf_path = os.path.join("data", "raw", "HSC26-Bangla1st-Paper.pdf")
    extract_from_pdf(pdf_path, os.path.join("data", "processed", "images"))