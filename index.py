import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers  import SentenceTransformer
from loader import load_papers
from chunking import token_chunks
import time

CHUNKER_NAME = "token_400_80"
SIZE = 400
OVERLAP = 80

if __name__ == "__main__":
    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    conn.execute("CREATE TABLE IF NOT EXISTS chunks ( id SERIAL PRIMARY KEY, paper_id TEXT NOT NULL, chunker TEXT NOT NULL, chunk_index INT NOT NULL, char_start INT NOT NULL, char_end INT NOT NULL, content TEXT NOT NULL, embedding VECTOR(384) NOT NULL)")

    conn.execute("CREATE INDEX IF NOT EXISTS chunks_paper_chunker_idx ON chunks (paper_id, chunker)")

    conn.execute("DELETE FROM chunks WHERE chunker = %s", (CHUNKER_NAME,))

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    papers = load_papers()

    start_time = time.perf_counter()

    total_chunks = 0
    max_tokens = 0

    over_limit = 0

    for i, paper in enumerate(papers, start = 1):
        chunks = token_chunks(paper["paragraphs"], model.tokenizer, SIZE, OVERLAP)

        if not chunks:
            continue

        texts = [c[0] for c in chunks]
        vecs = model.encode(texts, normalize_embeddings=True)
        

        rows = []
        for idx, ((content, start , end), vec) in enumerate(zip(chunks, vecs)) : 
            rows.append((paper["id"], CHUNKER_NAME, idx, start, end, content, vec))

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (paper_id, chunker, chunk_index, char_start, char_end, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )

        for chunk in texts:
            n_tokens = len(model.tokenizer.encode(chunk))
            max_tokens = max(max_tokens, n_tokens)

            if n_tokens > 512:
                over_limit += 1

        total_chunks += len(chunks)

        if i % 25 == 0:
            print(f"Indexed {i}/{len(papers)} papers")

    elapsed = time.perf_counter() - start_time

    print(f"Total Chunks: {total_chunks}")
    print(f"Seconds Taken: {elapsed:.1f}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Chunks over 512 tokens: {over_limit}")
    conn.close()




