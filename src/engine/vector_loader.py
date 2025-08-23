import faiss
import pickle
import json

def load_vector_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata
