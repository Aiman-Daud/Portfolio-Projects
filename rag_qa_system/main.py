"""FastAPI service exposing the RAG system as a REST API."""
import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import answer

app = FastAPI(title="RAG Question-Answering API")
LOG_PATH = "logs/queries.jsonl"


class Question(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set")
    start = time.time()
    result = answer(q.question, q.top_k)
    latency = round(time.time() - start, 2)

    # Query logging: the basis for monitoring (latency, low-score retrievals, bad answers)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "question": q.question,
            "top_score": result["sources"][0]["score"] if result["sources"] else None,
            "latency_s": latency,
        }) + "\n")
    return {**result, "latency_s": latency}
