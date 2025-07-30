import os, pickle, time, re, faiss, numpy as np, httpx
from bs4 import BeautifulSoup
from trafilatura import extract
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/sebi_index.faiss"
META_PATH  = "data/sebi_chunks.pkl"   

model = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_clean(url: str, timeout=15) -> str:
    html = httpx.get(url, timeout=timeout).text

    main_text = extract(html, include_comments=False, include_tables=False)
    if not main_text:
        soup = BeautifulSoup(html, "html.parser")
        main_text = soup.get_text(" ")
    return main_text

def chunkify(text: str, max_chars=1500):
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, buff = [], ""
    for p in paragraphs:
        if len(buff) + len(p) < max_chars:
            buff += ("\n" + p)
        else:
            chunks.append(buff.strip()); buff = p
    if buff: chunks.append(buff.strip())
    return chunks

def embed_and_add(chunks, index, meta):
    vecs = model.encode(chunks).astype("float32")
    index.add(vecs)
    meta.extend(chunks)
    return index, meta

def ingest_url(url: str):
    clean_text   = fetch_clean(url)
    new_chunks   = chunkify(clean_text)
    print(f"✅ {url} → {len(new_chunks)} chunks")

    # load current index
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f: meta = pickle.load(f)

    index, meta = embed_and_add(new_chunks, index, meta)


    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f: pickle.dump(meta, f)

    print("Index updated & saved")

if __name__ == "__main__":
    url_list = [
        "https://www.bankbazaar.com/home-loan-interest-rate.html"
    ]
    for u in url_list:
        try:
            ingest_url(u)
            time.sleep(1)   # polite delay
        except Exception as e:
            print(f" {u} failed: {e}")
