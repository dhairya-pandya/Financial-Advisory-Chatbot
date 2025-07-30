import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("data/sebi_index.faiss")

with open("data/sebi_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def retrieve_chunks(query, top_k=5):
    q_embed = model.encode(query) 
    q_embed = np.array([q_embed]).astype("float32") 
    D, I = faiss_index.search(q_embed, top_k)
    return [chunks[i] for i in I[0]]
