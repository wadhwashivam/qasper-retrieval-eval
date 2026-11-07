import os
import ollama

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from loader import load_papers

from sentence_transformers import SentenceTransformer

CHUNKER_NAME = "token_400_80"
LLM_MODEL = "llama3.1:8b"

def retrieve(conn, model, paper_id, question, k=5, chunker= CHUNKER_NAME):
        
    q_vec = model.encode(question,normalize_embeddings=True)

    rows = conn.execute(
        """
        SELECT chunk_index, content, 1 - (embedding <=> %s) AS similarity
        FROM chunks
        WHERE paper_id = %s AND chunker = %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (q_vec, paper_id, chunker, q_vec, k),
    ).fetchall()

    return rows

INSTRUCTIONS = (
    "Answer the question using only the numbered context passages below. "
    "Cite the passages you used with their numbers in square brackets, like [1] or [2][4]. "
    "If the context does not contain the answer, say exactly: "
    "\"The provided context does not answer this question.\""
)

def build_prompt(question, chunks):
    context = ""

    for n, (idx, content, sim) in enumerate(chunks, start = 1):
        context += f"[{n}] {content}\n\n"

    return f"{INSTRUCTIONS}\n\nContext:\n{context}Question:  {question}\n\nAnswer:"

def generate(prompt):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]


if __name__ == "__main__":
    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    register_vector(conn)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    paper_id = load_papers(limit=1)[0]["id"]

    questions = [
        "which multilingual approaches do they compare with?",
        "what are the pivot-based baselines?",
        "which datasets did they experiment with?",
        "what language pairs are explored?",
    ]

    for question in questions:
        print("=" * 60)
        chunks = retrieve(conn, model, paper_id, question)
        print(f"Q: {question}\n")

        for idx, content, sim in chunks:
            print(f"chunk {idx}  {sim:.3f}  {content[:150]}...")

        prompt = build_prompt(question, chunks)
        answer = generate(prompt)
        print(f"\nAnswer:\n{answer}\n")

    conn.close()