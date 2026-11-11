import numpy as np
import csv

def load(split, config):
    with open(f"results/{split}/per_question_{config}.csv") as f:
        rows = list(csv.DictReader(f))

    result = {(q["question_id"]): q for q in rows}
    return result

def gain_b1k(row):
    return float(row["hit_b1k"]) - float(row["rand_hit_b1k"])

def hit1(row):
    return float(row["hit1"])

def paired_diffs(a,b, metric):
    common = sorted(a.keys() & b.keys())
    return np.array([metric(a[q]) - metric(b[q]) for q in common])

def bootstrap(diffs, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return diffs.mean(), low, high, min(p, 1.0)


if __name__ == "__main__":
    comparisons = [
        ("para_400 vs token_400 (dense)",  "para_400_dense",  "token_400_80_dense",  gain_b1k),
        ("para_400 vs token_400 (hybrid)", "para_400_hybrid", "token_400_80_hybrid", gain_b1k),
        ("para_200 vs token_200 (dense)",  "para_200_dense",  "token_200_40_dense",  gain_b1k),
        ("para_200 vs token_200 (hybrid)", "para_200_hybrid", "token_200_40_hybrid", gain_b1k),
        ("hybrid vs dense (token_400)",    "token_400_80_hybrid", "token_400_80_dense", hit1),
        ("hybrid vs dense (para_400)",     "para_400_hybrid",     "para_400_dense",     hit1),
        ("rerank vs dense (token_400)",    "token_400_80_rerank", "token_400_80_dense", hit1),
        ("rerank vs dense (para_200)",     "para_200_rerank",     "para_200_dense",     hit1),
    ]

    for label, a_cfg, b_cfg, metric in comparisons:
        a = load("test", a_cfg)
        b = load("test", b_cfg)
        diffs = paired_diffs(a, b, metric)
        mean, low, high, p = bootstrap(diffs)
        verdict = "significant" if low > 0 or high < 0 else "not significant"
        print(f"{label:34} {mean*100:+6.1f} pts   [{low*100:+5.1f}, {high*100:+5.1f}]   "
              f"p={p:.3f}  n={len(diffs)}  {verdict}")

