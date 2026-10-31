import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)


conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)

conn.execute("DROP TABLE IF EXISTS test_items")
conn.execute("CREATE TABLE test_items ( id SERIAL PRIMARY KEY, content TEXT NOT NULL, embedding VECTOR(384) NOT NULL)")


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

documents = [
    "Cricket is a bat-and-ball game played between two teams of eleven players. "
        "A bowler delivers the ball toward the batter, who tries to hit it and score runs "
        "by running between the wickets. Test matches can last up to five days.",
    
        "Baseball is played between two teams of nine players over nine innings. "
        "The pitcher throws the ball to the batter, who tries to hit it and run around "
        "four bases. A run is scored when a player returns safely to home plate.",
    
        "Sourdough bread rises using a starter, a mix of flour and water that contains "
        "wild yeast and lactic acid bacteria. The bacteria give the bread its sour taste, "
        "and the slow fermentation can take twelve hours or more.",
    
        "A black hole forms when a massive star collapses under its own gravity. "
        "Its gravity is so strong that nothing, not even light, can escape once it "
        "crosses the boundary called the event horizon.",
    
        "Adults generally need seven to nine hours of sleep per night. During deep sleep "
        "the body repairs tissue, and during REM sleep the brain consolidates memories. "
        "Regular sleep deprivation is linked to weaker focus and immune function.",
]

doc_vecs = model.encode(documents, normalize_embeddings=True)

for content,vec in zip(documents, doc_vecs):
    conn.execute(
        "INSERT INTO test_items (content, embedding) VALUES (%s, %s)",
        (content, vec),
    )

question = "why am I forgetful after pulling an all-nighter?"
q_vec = model.encode(question, normalize_embeddings=True)

rows = conn.execute(
    """
    SELECT content, 1 - (embedding <=> %s) AS similarity
    FROM test_items
    ORDER BY embedding <=> %s
    LIMIT 2
    """,
    (q_vec, q_vec),
).fetchall()

print(f"Q: {question}")
for content, similarity in rows:
    print(f"    {similarity:.3f}  {content[:60]}...")

conn.close()