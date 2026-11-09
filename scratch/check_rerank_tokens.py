import os
import psycopg
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()
conn = psycopg.connect(os.environ["DATABASE_URL"])
reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

for chunker in ["token_400_80", "para_200"]:
    contents = [r[0] for r in conn.execute(
        "SELECT content FROM chunks WHERE chunker = %s", (chunker,)
    ).fetchall()]
    lengths = [len(reranker.tokenizer(c, add_special_tokens=False)["input_ids"]) for c in contents]
    over = sum(1 for n in lengths if n > 480)
    print(f"{chunker}: max {max(lengths)} reranker tokens, "
          f"{over} of {len(lengths)} chunks over 480 ({over / len(lengths):.1%})")

conn.close()