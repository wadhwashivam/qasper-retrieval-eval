import csv
import os
import random
import time
from statistics import mean

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from ask import retrieve
from labels import evidence_spans, relevant_chunks

from loader import load_papers, load_questions

CHUNKER= "token_400_80"
RETRIEVER = "dense"
KS = [1, 3, 5]
RANDOM_TRIALS = 200
SEED = 42

def first_relevant_rank(ranked, relevant):

    for pos,idx in enumerate(ranked, start = 1):
        if idx in relevant:
            return pos

    return None


def random_baseline(chunks_ids, relevant, rng):
    ids = list(chunks_ids)
    hits = {k:0 for k in KS}

    rr_total = 0.0

    for _ in range(RANDOM_TRIALS):
        rng.shuffle(ids)
        rank = first_relevant_rank(ids, relevant)
        for k in KS:
            if rank <= k:
                hits[k] += 1
        rr_total += 1/rank

    result = {f"rand_hit{k}": hits[k]/ RANDOM_TRIALS for k in KS}
    result["rand_rr"] = rr_total / RANDOM_TRIALS

    return result
    

if __name__ == "__main__":
    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    register_vector(conn)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    rng = random.Random(SEED)

    papers = {p["id"]: p for p in load_papers()}

    questions = load_questions()

    rows = []

    skipped_not_found = 0
    skipped_no_chunk = 0

    start_time = time.perf_counter()

    for q in questions:
        paper = papers[q["paper_id"]]

        spans = evidence_spans(paper["text"], q["evidence"])
        if not spans:
            skipped_not_found += 1 
            continue

        chunks = conn.execute(
                "SELECT chunk_index, char_start, char_end FROM chunks "
                "WHERE paper_id = %s AND chunker = %s ORDER BY chunk_index",
                (q["paper_id"], CHUNKER),
            ).fetchall()

        relevant = relevant_chunks(chunks,spans)
        if not relevant:
            skipped_no_chunk += 1
            continue

        results = retrieve(conn, model, q["paper_id"], q["question"], k=len(chunks), chunker=CHUNKER)
        ranked = [r[0] for r in results]
        rank = first_relevant_rank(ranked, relevant)

        row = {
            "question_id": q["question_id"],
            "paper_id": q["paper_id"],
            "question": q["question"],
            "n_chunks": len(chunks),
            "n_relevant": len(relevant),
            "rank": rank,
            "hit1": 1 if rank <=1 else 0,
            "hit3": 1 if rank <=3 else 0,
            "hit5": 1 if rank <=5 else 0,
            "rr": 1/rank,
        }

        row.update(random_baseline([c[0] for c in chunks], relevant, rng))

        rows.append(row)

    elapsed = time.perf_counter() - start_time

    summary = {
        "chunker": CHUNKER,
        "retriever": RETRIEVER,
        "questions": len(rows),
        "skipped_not_found": skipped_not_found,
        "skipped_no_chunk": skipped_no_chunk,
        "hit1": round(mean(r["hit1"] for r in rows), 3),
        "hit3": round(mean(r["hit3"] for r in rows), 3),
        "hit5": round(mean(r["hit5"] for r in rows), 3),
        "mrr": round(mean(r["rr"] for r in rows), 3),
        "rand_hit1": round(mean(r["rand_hit1"] for r in rows), 3),
        "rand_hit3": round(mean(r["rand_hit3"] for r in rows), 3),
        "rand_hit5": round(mean(r["rand_hit5"] for r in rows), 3),
        "rand_mrr": round(mean(r["rand_rr"] for r in rows), 3),
        "seconds": round(elapsed, 1),
    }
 
    print(f"\n{CHUNKER} + {RETRIEVER}")
    print(f"Questions evaluated: {summary['questions']} "
          f"(skipped: {skipped_not_found} evidence not found, {skipped_no_chunk} no covering chunk)")
    print(f"{'':10}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    print(f"{'Dense':10}{summary['hit1']:>8}{summary['hit3']:>8}{summary['hit5']:>8}{summary['mrr']:>8}")
    print(f"{'Random':10}{summary['rand_hit1']:>8}{summary['rand_hit3']:>8}"
          f"{summary['rand_hit5']:>8}{summary['rand_mrr']:>8}")
    print(f"Seconds: {summary['seconds']}")
 
    os.makedirs("results", exist_ok=True)
 
    # One row per question, for the Phase 4 failure analysis.
    with open(f"results/per_question_{CHUNKER}_{RETRIEVER}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
 
    # One summary row per run, appended, so every configuration ends up in one table.
    summary_path = "results/summary.csv"
    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)
 
    conn.close()
 


