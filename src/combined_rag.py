# -*- coding: utf-8 -*-
"""
এই স্ক্রিপ্টটি একটি উন্নত RAG (Retrieval-Augmented Generation) পাইপলাইন প্রয়োগ করে।
এটি Google-এর Gemini মডেল (জেনারেশনের জন্য) এবং OpenAI-এর মডেল (এমবেডিং-এর জন্য) ব্যবহার করে।
Pinecone ও AstraDB উভয় ভেক্টর স্টোর থেকে প্রাসঙ্গিক তথ্য সংগ্রহ করে ব্যবহারকারীর প্রশ্নের উত্তর দেয়।

বৈশিষ্ট্য:
- জেনারেশনের জন্য Google-এর শক্তিশালী 'gemini-1.5-pro-latest' মডেল ব্যবহার করে।
- এমবেডিং এর জন্য OpenAI-এর 'text-embedding-3-large' মডেল ব্যবহার করে।
- সরাসরি প্রশ্নের এমবেডিং ব্যবহার করে উভয় ডাটাবেস থেকে তথ্য পুনরুদ্ধার করে, যা নির্ভুলতা বাড়ায়।
- LangChain Expression Language (LCEL) ব্যবহার করে একটি মডুলার এবং সহজে বোধগম্য পাইপলাইন তৈরি করে।
- ডিবাগ মোড, যা পুনরুদ্ধার করা কনটেক্সট প্রদর্শন করে।
"""

import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
# FIX: Google GenAI এবং OpenAI উভয় লাইব্রেরি ইম্পোর্ট করা হয়েছে
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate

# --- ধাপ ১: এনভায়রনমেন্ট ভেরিয়েবল এবং ক্লায়েন্ট লোড করুন ---

# .env ফাইল থেকে API কী লোড করে
load_dotenv()

# --- কনফিগারেশন ---
PINECONE_INDEX_NAME = "combined-rag-index"
ASTRA_COLLECTION_NAME = "combined_rag_db"
# FIX: এমবেডিং মডেল OpenAI এবং জেনারেশন মডেল Gemini সেট করা হয়েছে
EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gemini-2.5-pro"

# Google, OpenAI, Pinecone, এবং AstraDB ক্লায়েন্ট ইনস্ট্যানশিয়েট করুন
try:
    # Google GenAI ক্লায়েন্ট (LLM এর জন্য)
    # FIX: .env ফাইল অনুযায়ী সঠিক কী এবং ভেরিয়েবলের নাম ব্যবহার করা হচ্ছে
    google_api_key = os.getenv("GEMINI_API_KEY_V2")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY_V2 এনভায়রনমেন্ট ভেরিয়েবল সেট করা নেই।")
    llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL, google_api_key=google_api_key, temperature=0.0)

    # OpenAI ক্লায়েন্ট (এমবেডিং এর জন্য)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY এনভায়রনমেন্ট ভেরিয়েবল সেট করা নেই।")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Pinecone ক্লায়েন্ট
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # AstraDB ভেক্টর স্টোর
    astradb_vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=ASTRA_COLLECTION_NAME,
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    print("সমস্ত ক্লায়েন্ট সফলভাবে ইনিশিয়ালাইজ করা হয়েছে।")
except Exception as e:
    print(f"ক্লায়েন্ট ইনিশিয়ালাইজেশনে ত্রুটি: {e}")
    exit()

# --- ধাপ ২: রিট্রিভার ফাংশন সংজ্ঞায়িত করুন ---

def retrieve_from_pinecone(query: str, top_k: int = 15) -> str:
    """সরাসরি প্রশ্নের এমবেডিং ব্যবহার করে Pinecone থেকে ডকুমেন্ট পুনরুদ্ধার করে।"""
    print(f"\n[Pinecone] {top_k}টি ডকুমেন্টের জন্য পুনরুদ্ধার প্রক্রিয়া শুরু...")
    try:
        query_embedding = embeddings.embed_query(query)
        results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = results.get("matches", [])
        context = "\n\n".join([match["metadata"]["text"] for match in matches])
        print(f"[Pinecone] {len(matches)}টি ডকুমেন্ট পাওয়া গেছে।")
        return context
    except Exception as e:
        print(f"[Pinecone] অনুসন্ধানের সময় ত্রুটি: {e}")
        return ""

def retrieve_from_astradb(query: str, top_k: int = 15) -> str:
    """AstraDB থেকে প্রাসঙ্গিক ডকুমেন্ট পুনরুদ্ধার করে।"""
    print(f"[AstraDB] {top_k}টি ডকুমেন্টের জন্য পুনরুদ্ধার প্রক্রিয়া শুরু...")
    try:
        results = astradb_vstore.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in results])
        print(f"[AstraDB] {len(results)}টি ডকুমেন্ট পাওয়া গেছে।")
        return context
    except Exception as e:
        print(f"[AstraDB] অনুসন্ধানের সময় ত্রুটি: {e}")
        return ""

# --- ধাপ ৩: RAG চেইন তৈরি করুন ---

def format_and_combine_context(docs_dict: dict) -> str:
    """সমান্তরালভাবে প্রাপ্ত কনটেক্সট একত্রিত করে এবং ডুপ্লিকেট বাদ দেয়।"""
    pinecone_context = docs_dict.get("pinecone", "")
    astradb_context = docs_dict.get("astradb", "")

    all_chunks = pinecone_context.split('\n\n') + astradb_context.split('\n\n')
    unique_chunks = list(dict.fromkeys(filter(None, all_chunks)))
    
    print(f"\n[Formatter] মোট {len(unique_chunks)}টি ইউনিক ডকুমেন্ট একত্রিত করা হয়েছে।")
    return "\n\n".join(unique_chunks)

# FIX: আরও উন্নত এবং সুস্পষ্ট প্রম্পট টেমপ্লেট
final_prompt_template = """
You are a meticulous Bengali literary analyst. Your task is to answer the user's question based *only* on the provided context from a story.

Follow these steps precisely:
1.  **Identify Evidence:** First, silently review the entire 'Context from documents' and identify the exact sentence or sentences that contain the answer to the 'Question'.
2.  **Analyze Evidence:** Pay close attention to the specific details. For questions about a person, make sure the description is about them and not someone else mentioned nearby. For questions involving time (like "at the time of the wedding"), find the sentence that matches that specific context.
3.  **Formulate Answer:** Based *only* on the evidence you identified, formulate a direct and concise answer in Bengali.
4.  **Handle No Evidence:** If you cannot find any sentence that directly answers the question after a thorough search, and only in that case, you *must* respond with: 'প্রদত্ত তথ্যে এই প্রশ্নের উত্তর খুঁজে পাওয়া যায়নি।'

Context from documents:
---
{context}
---

Question: {question}

Answer:
"""

final_prompt = PromptTemplate(
    template=final_prompt_template,
    input_variables=["context", "question"],
)

# রিট্রিভার চেইন, যা সমান্তরালভাবে উভয় ডাটাবেস থেকে তথ্য সংগ্রহ করে
retriever_chain = RunnableParallel(
    pinecone=RunnableLambda(retrieve_from_pinecone),
    astradb=RunnableLambda(retrieve_from_astradb)
)

# --- ধাপ ৪: প্রধান এক্সিকিউশন ব্লক ---

def parse_arguments():
    """ডিবাগ মোড চালু করার জন্য কমান্ড-লাইন আর্গুমেন্ট পার্স করে।"""
    import argparse
    parser = argparse.ArgumentParser(description="Run the RAG pipeline with optional debug mode.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, prints the retrieved context for each query before generating the answer.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # চূড়ান্ত RAG চেইন
    retrieval_and_formatting_chain = RunnableParallel({
        "context": retriever_chain | RunnableLambda(format_and_combine_context),
        "question": RunnablePassthrough()
    })

    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "শম্ভুনাথ বাবু পেশায় কী ছিলেন?",
    ]

    for query in test_queries:
        print("=" * 70)
        print(f"প্রশ্ন: {query}\n")
        
        # প্রথমে কনটেক্সট সংগ্রহ করুন
        retrieved_data = retrieval_and_formatting_chain.invoke(query)
        retrieved_context = retrieved_data["context"]

        # ডিবাগ মোড চালু থাকলে কনটেক্সট প্রিন্ট করুন
        if args.debug:
            print("\n" + "--- [DEBUG MODE: RETRIEVED CONTEXT] " + "-"*35)
            print(retrieved_context)
            print("--- [END OF DEBUG CONTEXT] " + "-"*41 + "\n")

        # এখন কনটেক্সট ব্যবহার করে চূড়ান্ত উত্তর তৈরি করুন
        generation_chain = final_prompt | llm
        final_answer = generation_chain.invoke(retrieved_data)
        
        print("\n" + "-"*30)
        print(f"চূড়ান্ত উত্তর:\n{final_answer.content.strip()}")
        print("-" * 30)
        print("=" * 70 + "\n\n")
