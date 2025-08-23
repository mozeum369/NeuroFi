import os
import re
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# Scrape AllCryptoWhitepapers.com
def scrape_allcrypto_whitepapers():
    url = "https://allcryptowhitepapers.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    entries = soup.select("article h2 a")
    documents = []
    for entry in entries[:10]:  # limit to top 10 for demo
        title = entry.text.strip()
        link = entry['href']
        documents.append({
            "text": title,
            "source": "AllCryptoWhitepapers.com",
            "url": link,
            "category": "Whitepaper"
        })
    return documents

# Scrape GitHub trending blockchain repos
def scrape_github_trending():
    url = "https://github.com/trending?since=daily&spoken_language_code=en"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    entries = soup.select("article.Box-row h2 a")
    documents = []
    for entry in entries[:10]:
        repo = entry.text.strip().replace('\n', '').replace(' ', '')
        link = "https://github.com" + entry['href']
        documents.append({
            "text": f"Trending GitHub repo: {repo}",
            "source": "GitHub",
            "url": link,
            "category": "Open Source"
        })
    return documents

# Aggregate and clean documents
def aggregate_documents():
    docs = []
    for source_func in [scrape_allcrypto_whitepapers, scrape_github_trending]:
        docs.extend(source_func())
    for doc in docs:
        doc["cleaned_text"] = clean_text(doc["text"])
    return docs

# Embed and index documents
def embed_and_index(documents):
    corpus = [doc["cleaned_text"] for doc in documents]
    embeddings = model.encode(corpus)
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.add(embeddings)
    return index, embeddings

# Save index and metadata
def save_index_and_metadata(index, documents):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_filename = f"neurofi_vector_index_{timestamp}.faiss"
    doc_filename = f"neurofi_documents_{timestamp}.json"
    faiss.write_index(index, index_filename)
    with open(doc_filename, "w") as f:
        json.dump(documents, f, indent=2)
    return index_filename, doc_filename

# Run the full pipeline
documents = aggregate_documents()
index, embeddings = embed_and_index(documents)
index_file, doc_file = save_index_and_metadata(index, documents)

print(f"Scraped and embedded documents saved to:\n- Index: {index_file}\n- Metadata: {doc_file}")


