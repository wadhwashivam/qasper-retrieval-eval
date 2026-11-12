from significance import load

configs = [
        "token_400_80_dense", "token_400_80_hybrid", "token_400_80_rerank",
        "token_200_40_dense", "token_200_40_hybrid",
        "para_400_dense", "para_400_hybrid",
        "para_200_dense", "para_200_hybrid", "para_200_rerank",
]

miss_sets = []

for cfg in configs:
    rows = load("test", cfg)
    misses = {qid for qid, row in rows.items() if row["hit_b1k"] == "0"}
    miss_sets.append(misses)

hard = set.intersection(*miss_sets)
total = len(load("test", "para_200_dense"))
print(f"Hard questions (missed by all {len(configs)} configs at ~1k): "
    f"{len(hard)} of {total} ({len(hard) / total:.1%})")

if __name__ == "__main__":
    import os
    import random
    import psycopg
    from dotenv import load_dotenv
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer
    from loader import load_papers, load_questions
    from retrievers import fetch_paper_chunks, dense_rank

    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    papers = {p["id"]: p for p in load_papers("test")}
    questions = {q["question_id"]: q for q in load_questions("test")}
    best = load("test", "para_200_dense")

    sample = random.Random(42).sample(sorted(hard), 20)

    with open("results/test/failures.md", "w") as f:
        for n, qid in enumerate(sample, start=1):
            q = questions[qid]
            chunks = fetch_paper_chunks(conn, q["paper_id"], "para_200")
            ranked = dense_rank(model, q["question"], chunks)
            text_by_idx = {c[0]: c[3] for c in chunks}

            f.write(f"## {n}. {q['question']}\n\n")
            f.write(f"Paper: {q['paper_id']}. Evidence rank with para_200 + dense: "
                    f"{best[qid]['rank']} of {best[qid]['n_chunks']}\n\n")
            for i, ev in enumerate(q["evidence"], start=1):
                f.write(f"**Gold evidence {i}:** {ev[:700]}\n\n")
            for pos in range(2):
                f.write(f"**Retrieved #{pos + 1}:** {text_by_idx[ranked[pos]][:700]}\n\n")
            f.write("**Category:** \n\n---\n\n")

    conn.close()
    print("Wrote results/test/failures.md")