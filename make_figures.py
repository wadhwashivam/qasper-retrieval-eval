import csv
import os

import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# Figure 1: gain over random at ~1k tokens, test split
with open("results/test/summary.csv") as f:
    rows = list(csv.DictReader(f))
latest = {(r["chunker"], r["retriever"]): r for r in rows}

chunkers = ["token_400_80", "para_400", "token_200_40", "para_200"]
labels = ["Fixed 400", "Paragraph 400", "Fixed 200", "Paragraph 200"]
retrievers = ["dense", "hybrid", "rerank"]

x = np.arange(len(chunkers))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, r in enumerate(retrievers):
    gains = []
    for c in chunkers:
        row = latest.get((c, r))
        gains.append(float(row["hit_b1k"]) - float(row["rand_hit_b1k"]) if row else np.nan)
    ax.bar(x + (i - 1) * width, gains, width, label=r)
ax.set_xticks(x, labels)
ax.set_ylabel("Gain over random\n(evidence within ~1k tokens)")
ax.set_title("Evidence retrieval on QASPER test set (n = 1,308)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/gain_by_config.png", dpi=200)
plt.close(fig)

# Figure 2: paired bootstrap differences with 95% intervals
with open("results/test/significance.csv") as f:
    sig = list(csv.DictReader(f))

names = [r["label"] for r in sig]
mean = np.array([float(r["mean"]) for r in sig]) * 100
low = np.array([float(r["low"]) for r in sig]) * 100
high = np.array([float(r["high"]) for r in sig]) * 100
y = np.arange(len(sig))[::-1]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="o", capsize=4)
ax.axvline(0, color="gray", linestyle="--")
ax.set_yticks(y, names)
ax.set_xlabel("Difference (percentage points), 95% bootstrap interval")
ax.set_title("Paired comparisons on the test set")
fig.tight_layout()
fig.savefig("figures/significance.png", dpi=200)
plt.close(fig)

print("Saved figures/gain_by_config.png and figures/significance.png")