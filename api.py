import os
from collections import defaultdict
from contextlib import asynccontextmanager

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ask import build_prompt, generate
from loader import load_papers, load_questions
from retrievers import fetch_paper_chunks, dense_search

CHUNKER = "para_200"
TOP_K = 5
state = {}


@asynccontextmanager
async def lifespan(app):
    load_dotenv()
    state["model"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    state["titles"] = {p["id"]: p["title"] for p in load_papers("validation")}
    questions = defaultdict(list)
    for q in load_questions("validation"):
        questions[q["paper_id"]].append(q["question"])
    state["questions"] = questions
    yield


app = FastAPI(title="QASPER RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    paper_id: str
    question: str = Field(min_length=3, max_length=500)


class Passage(BaseModel):
    number: int
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    passages: list[Passage]

def get_conn():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/papers")
def list_papers():
    return [{"id": pid, "title": title} for pid, title in state["titles"].items()]

@app.get("/papers/{paper_id}/questions")
def paper_questions(paper_id: str):
    if paper_id not in state["titles"]:
        raise HTTPException(status_code=404, detail="Paper not found")
    return state["questions"][paper_id]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.paper_id not in state["titles"]:
        raise HTTPException(status_code=404, detail= "Paper not found")

    with get_conn() as conn:
        chunks = fetch_paper_chunks(conn, req.paper_id, CHUNKER)

    results = dense_search(state["model"], req.question, chunks, TOP_K )
    prompt = build_prompt(req.question, results)


    try:
        answer = generate(prompt)
    except Exception:
        raise HTTPException(status_code=503, detail="LLM unavailable. Is Ollama running?")

    passages = [
        {"number": n, "chunk_index": idx, "score": score, "text": text}
        for n, (idx, text, score) in enumerate(results, start = 1)
    ]

    return {"answer": answer, "passages": passages}