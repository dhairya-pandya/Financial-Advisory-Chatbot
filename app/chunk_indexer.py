import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

def chunk_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            for para in text.split('\n\n'):
                if len(para.strip()) > 80:
                    chunks.append(para.strip())
    return chunks

def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def save_index(index, chunks):
    faiss.write_index(index, "data/sebi_index.faiss")
    with open("data/sebi_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    chunks = chunk_pdf("data/sebi_document.pdf")
    index, _ = build_faiss_index(chunks)
    save_index(index, chunks)
    print(f"Indexed {len(chunks)} chunks.")
