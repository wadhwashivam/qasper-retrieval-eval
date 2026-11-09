import csv

with open("results/summary.csv") as f:
    rows = list(csv.DictReader(f))

latest = {(r["chunker"], r["retriever"]): r for r in rows}

chunkers = ["token_400_80", "token_200_40", "para_400", "para_200"]
retrievers = ["dense", "bm25", "hybrid", "rerank"]

print("| Chunker | Retriever | Hit@1 | MRR | Budget ~1k | Gain ~1k | Budget ~2k | Gain ~2k | Seconds |")
print("|---|---|---|---|---|---|---|---|---|")
for c in chunkers:
    for r in retrievers:
        row = latest.get((c, r))
        if row is None:
            continue
        gain1 = float(row["hit_b1k"]) - float(row["rand_hit_b1k"])
        gain2 = float(row["hit_b2k"]) - float(row["rand_hit_b2k"])
        print(f"| {c} | {r} | {row['hit1']} | {row['mrr']} | {row['hit_b1k']} | {gain1:.3f} "
              f"| {row['hit_b2k']} | {gain2:.3f} | {row['seconds']} |")