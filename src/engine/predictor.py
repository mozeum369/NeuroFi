# predictor.py

from .vector_loader import load_vector_index
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model and index once
model = SentenceTransformer('all-MiniLM-L6-v2')
index, metadata = load_vector_index('data/faiss_index.idx', 'data/metadata.pkl')

def semantic_search(query, top_k=5):
    query_vector = model.encode([query])
    _, indices = index.search(np.array(query_vector), top_k)
    return [metadata[i] for i in indices[0]]

def run_model_with_context(context_docs):
    # Placeholder for your actual prediction logic
    # You can replace this with your ML model or rule-based engine
    combined_context = " ".join(doc['text'] for doc in context_docs)
    return {
        "context_used": combined_context,
        "prediction": "Surge likely due to social sentiment and low liquidity."
    }

def enhanced_prediction(query):
    context_docs = semantic_search(query)
    return run_model_with_context(context_docs)
