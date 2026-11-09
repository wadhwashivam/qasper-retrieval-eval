import csv
import os
import random
import time
from statistics import mean

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from retrievers import fetch_paper_chunks, dense_rank, bm25_rank, hybrid_rank
from labels import evidence_spans, relevant_chunks

from loader import load_papers, load_questions

import sys

CHUNKER= sys.argv[1] if len(sys.argv) > 1 else "token_400_80"
RETRIEVER = sys.argv[2] if len(sys.argv) > 2 else "dense"
KS = [1, 3, 5]
BUDGETS = {"b1k": 4750, "b2k": 9500}
RANDOM_TRIALS = 200
SEED = 42

def first_relevant_rank(ranked, relevant):

    for pos,idx in enumerate(ranked, start = 1):
        if idx in relevant:
            return pos

    return None


def random_baseline(chunks_ids, relevant, rng, lengths):
    ids = list(chunks_ids)
    hits = {k:0 for k in KS}
    budget_hits = {name: 0 for name in BUDGETS}

    rr_total = 0.0

    for _ in range(RANDOM_TRIALS):
        rng.shuffle(ids)
        rank = first_relevant_rank(ids, relevant)
        for k in KS:
            if rank <= k:
                hits[k] += 1
        rr_total += 1/rank

        for name, budget in BUDGETS.items():
            budget_hits[name] += hit_within_budget(ids, relevant, lengths, budget)

    result = {f"rand_hit{k}": hits[k]/ RANDOM_TRIALS for k in KS}
    result["rand_rr"] = rr_total / RANDOM_TRIALS

    for name in BUDGETS:
        result[f"rand_hit_{name}"] = budget_hits[name] / RANDOM_TRIALS

    return result


def hit_within_budget(ranked, relevant, lengths, budget):
    used = 0
    for chunk_index in ranked:
        if used >= budget:
            return 0
        if chunk_index in relevant:
            return 1
        used += lengths[chunk_index]
    return 0        

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

        chunks = fetch_paper_chunks(conn,q["paper_id"], CHUNKER)

        relevant = relevant_chunks([c[:3] for c in chunks], spans)

        if not relevant:
            skipped_no_chunk += 1
            continue

        if RETRIEVER == "dense":
            ranked = dense_rank(model, q["question"], chunks)
        elif RETRIEVER == "bm25":
            ranked = bm25_rank(q["question"], chunks)
        elif RETRIEVER == "hybrid":
            ranked = hybrid_rank(model, q["question"], chunks)
        else:
            raise ValueError(f"Unknown retriever: {RETRIEVER}")
        
        rank = first_relevant_rank(ranked, relevant)

        lengths = {c[0]: c[2] - c[1] for c in chunks}

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
            "hit_b1k": hit_within_budget(ranked, relevant, lengths, BUDGETS["b1k"]),
            "hit_b2k": hit_within_budget(ranked, relevant, lengths, BUDGETS["b2k"]),
        }

        row.update(random_baseline([c[0] for c in chunks], relevant, rng, lengths))

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
        "hit_b1k": round(mean(r["hit_b1k"] for r in rows), 3),
        "hit_b2k": round(mean(r["hit_b2k"] for r in rows), 3),
        "rand_hit_b1k": round(mean(r["rand_hit_b1k"] for r in rows), 3),
        "rand_hit_b2k": round(mean(r["rand_hit_b2k"] for r in rows), 3),
    }

    print(f"\n{CHUNKER} + {RETRIEVER}")
    print(f"Questions evaluated: {summary['questions']} "
          f"(skipped: {skipped_not_found} evidence not found, {skipped_no_chunk} no covering chunk)")
    print(f"{'':10}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    print(f"{RETRIEVER:10}{summary['hit1']:>8}{summary['hit3']:>8}{summary['hit5']:>8}{summary['mrr']:>8}")
    print(f"{'Random':10}{summary['rand_hit1']:>8}{summary['rand_hit3']:>8}"
          f"{summary['rand_hit5']:>8}{summary['rand_mrr']:>8}")
    print(f"Seconds: {summary['seconds']}")

    print(f"Budget ~1k tokens: {RETRIEVER} {summary['hit_b1k']} vs random {summary['rand_hit_b1k']}")
    print(f"Budget ~2k tokens: {RETRIEVER} {summary['hit_b2k']} vs random {summary['rand_hit_b2k']}")
 
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
 


