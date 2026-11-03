"""Core RAG logic: load documents -> chunk -> embed -> FAISS index -> retrieve -> LLM answer."""
import glob
import os
import pickle

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_PATH = "index/faiss.index"
CHUNKS_PATH = "index/chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"      # small, free, runs on CPU
LLM_MODEL = "openai/gpt-oss-20b"    # served free by Groq

_model = None
_index = None
_chunks = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def load_documents():
    """Return a list of (source_file, page_number, text) from PDFs and .txt files in data/."""
    docs = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*")):
        name = os.path.basename(path)
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append((name, i, text))
        elif path.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                docs.append((name, 1, f.read()))
    return docs


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_index():
    docs = load_documents()
    if not docs:
        raise RuntimeError("No PDF or .txt files found in data/. Add some first.")
    chunks = []
    for source, page, text in docs:
        for c in chunk_text(text):
            chunks.append({"source": source, "page": page, "text": c})

    embeddings = get_model().encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine on normalized vectors
    index.add(np.array(embeddings, dtype="float32"))

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages.")


def load_index():
    global _index, _chunks
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            build_index()
        _index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
    return _index, _chunks


def retrieve(question, k=3):
    index, chunks = load_index()
    q = get_model().encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


def answer(question, k=3):
    from groq import Groq

    hits = retrieve(question, k)
    context = "\n\n".join(
        f"[{h['source']}, page {h['page']}]\n{h['text']}" for h in hits
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            {"source": h["source"], "page": h["page"], "score": round(h["score"], 3)}
            for h in hits
        ],
    }
