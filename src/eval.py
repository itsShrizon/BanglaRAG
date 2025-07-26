from sklearn.metrics.pairwise import cosine_similarity
from combined_rag import get_embedding, combined_rag

def evaluate_groundedness(expected, actual):  
    return actual.strip() == expected.strip()

def evaluate_relevance(query, chunks):
    query_embedding = get_embedding(query)
    chunk_embeddings = [get_embedding(chunk["metadata"]["text"]) for chunk in chunks]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    return np.mean(similarities)

if __name__ == "__main__":
    import numpy as np  
    test_cases = [
        {"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শুম্ভুনাথ"},
        {"query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"}
    ]
    for case in test_cases:
        query = case["query"]  
        answer, chunks = combined_rag(query)
        grounded = evaluate_groundedness(case["expected"], answer)
        relevance = evaluate_relevance(query, chunks)
        print(f"Query: {query}\nGrounded: {grounded}\nRelevance: {relevance:.2f}\n")