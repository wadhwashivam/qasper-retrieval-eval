# RAG Project — Lab Notes

## Dataset: QASPER

### Validation split (dev set)
- 281 papers, 1,005 questions.
- Paragraphs per paper: avg 48.4, min 7, max 292. Papers are long enough that chunking matters.
- 328 empty paragraph strings across the split. The loader skips them.
- 1.76 annotator answers per question on average. A paragraph is relevant if *any* annotator cited it.
- 163 annotator answers marked unanswerable.
- Evidence strings: 2,808 total. 253 point at tables/figures ("FLOAT SELECTED"), 2,555 are text.
- 91.9% of text evidence (2,349 of 2,555) appears verbatim in `full_text`.
  - Mismatch causes seen: section headings cited as evidence (e.g. "Languages", "Datasets"), and spacing differences (e.g. "BIBREF20 , which").
- 892 of 1,005 questions have at least one text evidence paragraph. The other 113 are excluded from retrieval evaluation (no text evidence).
- After filtering (text evidence only, ≥5 words, deduplicated across annotators): **889 usable questions**. 3 questions were dropped because their only evidence was heading-length. 888 evaluated (1 skipped: evidence not locatable in the text).

### Test split (held out until Phase 4)
- 416 papers. **1,308 questions evaluated.**

## Design decisions

- Retrieval is scoped to the question's own paper (`WHERE paper_id = ...`). QASPER questions refer to "they"/"this paper", so cross-paper search would mix chunking quality with paper identification.
- Loader: abstract first, then all `full_text` paragraphs in order, empty ones removed, whitespace normalized with `" ".join(p.split())`.
- Embeddings: `BAAI/bge-small-en-v1.5`, 384 dims, normalized, so cosine similarity = dot product.
- Exact vector search (no HNSW index) so approximation noise doesn't affect results.
- Each chunk stores `char_start`/`char_end` positions in the paper's joined text (`" ".join(paragraphs)`). Relevance is computed by span overlap.
- **Relevance rule (symmetric):** a chunk is relevant if it covers ≥50% of an evidence paragraph's span, **or** if ≥50% of the chunk is evidence. The original one-directional rule (≥50% coverage only) excluded 19 questions for 200-token chunks, because small chunks can't cover half of a long evidence paragraph; changed in Phase 3a so all chunkers are graded on the same questions.
- Relevance rule validated on paper 1912.01214, Q "which multilingual approaches do they compare with?": evidence span (19730, 20675); chunk 13 covers 100%, chunk 14 covers 10% (overlap carry-over). Rule output {13} matches the manual diagnosis from Phase 1.
- Dev/test: QASPER validation split = dev (all comparisons and decisions); QASPER test split = held out, run once on pre-registered configurations in Phase 4.

## Finding: word-based chunks exceeded the model's token limit

- First chunker: fixed 200 words, 40-word overlap.
- On the first 5 papers (119 chunks), the longest chunk was **655 tokens**; bge truncates input at 512 tokens.
- Chunks over 512 tokens: 4 of 119.
- Cause: scientific text tokenizes heavily (LaTeX, BIBREF/TABREF placeholders, numbers) — over 3 tokens per word in dense passages.
- Consequence: the tail of long chunks would be silently dropped from embeddings, biasing retrieval results.
- Fix: switched to token-based chunking using the model's own tokenizer (sizes in model tokens), slicing the original text by token character offsets so casing and spacing are preserved.

## Index runs

| Chunker | Split | Papers | Chunks | Seconds | Max tokens |
|---|---|---|---|---|---|
| fixed_200_40 (words) | validation | 5 | 119 | 1.4 | 655 |
| token_400_80 | validation | 5 | 92 | 1.1 | 403 |
| token_400_80 | validation | 281 | 4,619 | 27.3 | 404 |
| token_200_40 | validation | 281 | TODO | TODO | TODO |
| para_400 | validation | 281 | 4,636 | 26.4 | 402 |
| para_200 | validation | 281 | 10,413 | 28.3 | 203 |
| all four chunkers | test | 416 | TODO | TODO | TODO |

## Phase 1: end-to-end pipeline

- Pipeline: `loader.py` → `chunking.py` → `index.py` (indexing time) and `ask.py` (query time).
- Generation: `llama3.1:8b` via Ollama, temperature 0 (deterministic, reproducible), top k = 5.
- Ollama default context window is 4,096 tokens and truncates silently. 5 chunks × ~400 tokens + instructions ≈ 2,500 tokens, so it fits. Raising k to 10 would exceed it.
- Prompt tells the model to answer only from the numbered passages, cite them as [n], and say "The provided context does not answer this question." if the answer is absent.

### First test: paper 1912.01214 (4 QASPER questions, token_400_80 + dense)

| Question | Result |
|---|---|
| which multilingual approaches do they compare with? | Refused. Retrieval failure (see failure case 1). |
| what are the pivot-based baselines? | Answered, cited [3] (chunk 3). |
| which datasets did they experiment with? | Refused. Generation failure: Europarl/MultiUN appear in chunks 10, 11, 16, all in the top 5 (chunk 11 ranked first), but the model still refused. Likely cause: strict refusal instruction makes the 8B model over-cautious when passages don't match the question's wording. |
| what language pairs are explored? | Refused. Not diagnosed. |

### Failure case 1: evidence split across chunks

- Question: "which multilingual approaches do they compare with?"
- Gold evidence is in chunk 13 (found with SQL `LIKE`), which ranked **10th of 19**, similarity 0.689.
- Cause: chunk 13 opens with training setup (masking strategy, BPE), and the evidence paragraph sits at its tail. The paragraph's last sentence spills into chunk 14 (rank 6) through the overlap. The fixed-size chunker split the evidence and mixed it with unrelated text, diluting the chunk's embedding.
- Hypothesis for Phase 3: a paragraph- or section-aware chunker keeps the evidence paragraph whole and should rank it higher.
- Re-test in Phase 3b: para_200 ranks the evidence **3rd** (vs 10th with token_400_80); found within ~1k tokens. para_400 rank: TODO.

### Observations

- **Flat scores:** all 19 chunks scored between 0.637 and 0.726 for this question. When a question uses the paper's own topic vocabulary ("multilingual"), dense retrieval can't separate the chunks.
- **Random baseline needed:** the paper has only 19 chunks, so top 5 covers ~26% of it. A random retriever would hit the evidence about a quarter of the time. Recall@k must be reported alongside a random baseline.
- **Two failure types:**
  - Retrieval failure: evidence not in the top k, so refusing is correct.
  - Generation failure: evidence in the top k, but the model refuses or misses it.
- Phase 2 evaluates retrieval alone, with no LLM, so chunker and retriever quality are measured without generation noise mixed in.
- **Strict refusal tradeoff:** the "say exactly" refusal rule prevents made-up answers but causes wrongful refusals with an 8B model. Candidate side experiment: softer wording or a larger model (Gemma 4).

## Phase 2 results: baseline (dev set)

| Config | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| token_400_80 + dense | 888 | 0.287 | 0.597 | 0.759 | 0.490 |
| token_400_80 + random | 888 | 0.131 | 0.349 | 0.525 | 0.314 |

- 889 usable questions; 1 skipped (evidence not locatable), 0 skipped for no covering chunk. Eval runs in ~14 s.
- Dense clearly beats random, but random Hit@5 is already 0.525 (~16 chunks per paper, so top 5 ≈ a third of the paper). Hit@1 and MRR are the more discriminating metrics.
- Dense ranks the gold evidence first only 29% of the time, so there is real headroom for Phase 3.
- Methodology issue for Phase 3: chunkers with different chunk counts have different random baselines. Compare at equal token budget, not equal k.
- Deterministic: two runs give identical numbers.

## Phase 3a: chunk size (dense retrieval, symmetric relevance rule, dev set)

| Chunker | Hit@1 | Hit@5 | MRR | Budget ~1k | Random ~1k | Budget ~2k | Random ~2k |
|---|---|---|---|---|---|---|---|
| token_400_80 | 0.287 | 0.761 | 0.490 | 0.616 | 0.364 | 0.809 | 0.592 |
| token_200_40 | 0.293 | 0.650 | 0.459 | 0.686 | 0.382 | 0.858 | 0.610 |

- Retrieval moved from SQL to Python (fetch a paper's chunks once, rank in numpy). Python dense ranking reproduces the SQL results exactly (0.287 / 0.597 / 0.759 / 0.490).
- Budget metric: walk down the ranked list, summing chunk lengths; hit if a relevant chunk starts before the budget runs out. Budgets 4,750 chars (~1k tokens) and 9,500 chars (~2k tokens).
- Budget metric validated: random scores are nearly equal across chunk sizes, unlike Hit@5 (random 0.527 vs 0.339).
- 200-token chunks beat 400-token chunks at equal text budget: +5.2 points over random at ~1k, +3.1 at ~2k.
- Hit@k and MRR are not comparable across chunk sizes; use them only to compare retrievers on the same chunker.

## Phase 3b: paragraph-aware chunking (dense retrieval, dev set)

| Chunker | Chunks | Hit@1 | MRR | Budget ~1k | Random ~1k | Gain ~1k | Budget ~2k | Random ~2k | Gain ~2k |
|---|---|---|---|---|---|---|---|---|---|
| token_400_80 | 4,619 | 0.287 | 0.490 | 0.616 | 0.364 | 0.252 | 0.809 | 0.592 | 0.217 |
| para_400 | 4,636 | 0.310 | 0.495 | 0.664 | 0.382 | 0.282 | 0.848 | 0.626 | 0.222 |
| token_200_40 | TODO | 0.293 | 0.459 | 0.686 | 0.382 | 0.304 | 0.858 | 0.610 | 0.248 |
| para_200 | 10,413 | 0.302 | 0.469 | 0.771 | 0.416 | 0.355 | 0.900 | 0.655 | 0.245 |

Gain = dense minus random at the same budget (the fair comparison across chunkers).

- Paragraph chunker: merge consecutive paragraphs up to the token limit; split any single paragraph over the limit with the token chunker; no overlap. Spans verified: every chunk's text equals its slice of the paper text.
- Paragraph-aware chunking beats fixed-token chunking at a ~1k-token budget (gain over random: +3.0 points at 400, +5.1 at 200), and raises Hit@1 (0.310 vs 0.287 at 400).
- At ~2k tokens the advantage disappears (gains within ~0.5 points): chunk boundaries matter most when the context budget is tight.
- Variable chunk lengths make random baselines differ across chunkers, so compare by gain over random.

## Phase 3c: BM25 and hybrid (RRF, k=60), dev set

### token_400_80
| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Budget ~1k | Budget ~2k | Seconds |
|---|---|---|---|---|---|---|---|
| dense | 0.287 | 0.598 | 0.761 | 0.490 | 0.616 | 0.809 | ~15 |
| bm25 | 0.242 | 0.546 | 0.708 | 0.444 | 0.553 | 0.759 | 3.1 |
| hybrid | 0.323 | 0.636 | 0.788 | 0.519 | 0.639 | 0.814 | 14.7 |

### para_200
| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Budget ~1k | Budget ~2k | Seconds |
|---|---|---|---|---|---|---|---|
| dense | 0.302 | 0.539 | 0.681 | 0.469 | 0.771 | 0.900 | ~16 |
| bm25 | 0.181 | 0.408 | 0.573 | 0.353 | 0.653 | 0.834 | 4.1 |
| hybrid | 0.280 | 0.523 | 0.681 | 0.454 | 0.752 | 0.900 | 15.5 |

- BM25 index built per question over that paper's chunks only (IDF is within-paper). Tokenizer: lowercase, `\w+` regex.
- On 400-token chunks, hybrid beat dense and BM25 on every metric on dev (Hit@1 0.287 → 0.323). **Did not replicate on test (see Phase 4).**
- On para_200, hybrid is worse than dense (Hit@1 0.302 → 0.280). BM25 is much weaker on short chunks (12-point Hit@1 gap vs 4.5 on 400-token chunks), and equal-weight RRF lets the weaker retriever drag the stronger one down.
- Weighted RRF is possible but would mean tuning on dev; skipped to avoid overfitting.

## Phase 3d: reranker (BAAI/bge-reranker-base, reranks all chunks in the paper), dev set

- token_400_80 + rerank: Hit@1 0.248, Hit@3 0.556, Hit@5 0.756, MRR 0.455, Budget ~1k 0.568, ~2k 0.791. 272.5 s (~18x slower than dense).
- Unexpected: worse than dense (Hit@1 0.287) and hybrid (0.323) on the same chunks. Investigated truncation first (the reranker uses a different tokenizer, XLM-RoBERTa).
- Truncation check: 113 of 4,619 token_400_80 chunks (2.4%) exceed 480 reranker tokens (max 518); para_200 has none (max 265). Too few to explain the drop; truncation mostly ruled out.
- para_200 + rerank (no truncation possible): Hit@1 0.190 (dense 0.302), MRR 0.369 (dense 0.469), Budget ~1k 0.707 (dense 0.771), Budget ~2k 0.869 (dense 0.900). 281.3 s.
- token_200_40 + rerank: Hit@1 0.193 (dense 0.293), MRR 0.369 (dense 0.459). 270.6 s.
- para_400 + rerank: Hit@1 0.233 (dense 0.310), MRR 0.444 (dense 0.495). 276.6 s. (First attempt crashed when the Postgres container stopped mid-run, likely memory pressure; re-run after restarting it.)
- Sanity check: Hit@1 is ~3x random, so the reranker works; it just ranks worse than dense here.
- Conclusion: the reranker's underperformance is real, not a truncation artifact. Worse on shorter chunks, where it resembles BM25. Hypothesis: domain mismatch (trained on web search; within-paper chunks are all on-topic) plus terse, context-dependent questions. To probe in Phase 4 failure analysis.
- Decision: dropped "hybrid + rerank" from the plan, since the reranker alone underperforms dense on all four chunkers.

## Phase 3: full grid (dev set, 888 questions)

| Chunker | Retriever | Hit@1 | MRR | Budget ~1k | Gain ~1k | Budget ~2k | Gain ~2k | Seconds |
|---|---|---|---|---|---|---|---|---|
| token_400_80 | dense | 0.287 | 0.490 | 0.616 | 0.252 | 0.809 | 0.217 | 14.5 |
| token_400_80 | bm25 | 0.242 | 0.444 | 0.553 | 0.189 | 0.759 | 0.167 | 3.1 |
| token_400_80 | hybrid | 0.323 | 0.519 | 0.639 | 0.275 | 0.814 | 0.222 | 14.7 |
| token_400_80 | rerank | 0.248 | 0.455 | 0.568 | 0.204 | 0.791 | 0.199 | 272.5 |
| token_200_40 | dense | 0.293 | 0.459 | 0.686 | 0.304 | 0.858 | 0.248 | 16.1 |
| token_200_40 | bm25 | 0.173 | 0.349 | 0.596 | 0.214 | 0.786 | 0.176 | 3.9 |
| token_200_40 | hybrid | 0.276 | 0.447 | 0.695 | 0.313 | 0.865 | 0.255 | 16.6 |
| token_200_40 | rerank | 0.193 | 0.369 | 0.622 | 0.240 | 0.811 | 0.201 | 270.6 |
| para_400 | dense | 0.310 | 0.495 | 0.664 | 0.282 | 0.848 | 0.222 | 14.4 |
| para_400 | bm25 | 0.230 | 0.432 | 0.604 | 0.222 | 0.806 | 0.180 | 2.7 |
| para_400 | hybrid | 0.332 | 0.516 | 0.670 | 0.288 | 0.850 | 0.224 | 15.1 |
| para_400 | rerank | 0.233 | 0.444 | 0.625 | 0.243 | 0.848 | 0.222 | 276.6 |
| para_200 | dense | 0.302 | 0.469 | 0.771 | 0.355 | 0.900 | 0.245 | 15.6 |
| para_200 | bm25 | 0.181 | 0.353 | 0.653 | 0.237 | 0.834 | 0.179 | 4.1 |
| para_200 | hybrid | 0.280 | 0.454 | 0.752 | 0.336 | 0.900 | 0.245 | 15.5 |
| para_200 | rerank | 0.190 | 0.369 | 0.707 | 0.291 | 0.869 | 0.214 | 281.3 |

Gain = retriever minus random at the same budget. Hit@1 and MRR are comparable only between retrievers on the same chunker; compare chunkers by gain.

### Dev-set findings (before the test split; see Phase 4 for what held up)
1. Paragraph-aware chunking beats fixed-token chunking with every retriever, at both sizes (gain at ~1k budget, 6 of 6 dense/bm25/hybrid comparisons). Confirms the chunk 13 hypothesis at scale.
2. Hybrid (RRF) beats dense with ~400-token chunks (token_400 Hit@1 0.287 → 0.323; para_400 0.310 → 0.332) but not with ~200-token chunks (token_200 0.293 → 0.276; para_200 0.302 → 0.280). **Refuted on test: hybrid ≈ dense (Phase 4b).**
3. The cross-encoder reranker underperformed dense on all four chunkers, at ~18x the runtime. Truncation ruled out.
4. BM25 alone is the weakest retriever everywhere, worst on short chunks.
5. Best at ~1k budget: para_200 + dense (gain 0.355 over random). Best Hit@1: para_400 + hybrid (0.332).
6. At the ~2k budget, gains converge (most dense/hybrid/rerank configs between 0.20 and 0.25); configurations separate mainly at the ~1k budget (0.19 to 0.355). Design choices matter most when the context budget is tight.

### Decisions
- Semantic chunker (3e) skipped: findings were already consistent across configurations, and Phase 4 (test split, significance, failure analysis) adds more value.

## Phase 4a: test split (pre-registered, written before running)

Configurations (chosen on dev, run once on test):
- dense and hybrid on all four chunkers: token_400_80, token_200_40, para_400, para_200 (8 runs)
- rerank on token_400_80 and para_200 (2 runs)

Expectations from dev, to be confirmed or refuted:
1. Paragraph chunking beats fixed-token chunking at equal size (gain at ~1k budget), for dense and hybrid.
2. Hybrid beats dense on Hit@1 with ~400-token chunks, but not with ~200-token chunks.
3. Rerank underperforms dense on both chunkers tested.
4. para_200 + dense has the highest gain at the ~1k budget.

Rule: no changes to code, chunkers or parameters after seeing test results. Results reported as-is. Code and pre-registration committed to Git before the test run.

### Test results (run once, reported as-is; 1,308 questions)

| Chunker | Retriever | Hit@1 | MRR | Budget ~1k | Gain ~1k | Budget ~2k | Gain ~2k | Seconds |
|---|---|---|---|---|---|---|---|---|
| token_400_80 | dense | 0.365 | 0.554 | 0.690 | 0.281 | 0.845 | 0.208 | 22.1 |
| token_400_80 | hybrid | 0.362 | 0.551 | 0.669 | 0.260 | 0.845 | 0.208 | 23.4 |
| token_400_80 | rerank | 0.310 | 0.509 | 0.638 | 0.229 | 0.822 | 0.185 | 402.1 |
| token_200_40 | dense | 0.317 | 0.492 | 0.751 | 0.328 | 0.888 | 0.239 | 24.7 |
| token_200_40 | hybrid | 0.292 | 0.471 | 0.723 | 0.300 | 0.887 | 0.238 | 26.3 |
| para_400 | dense | 0.357 | 0.542 | 0.720 | 0.300 | 0.881 | 0.218 | 22.0 |
| para_400 | hybrid | 0.352 | 0.538 | 0.713 | 0.293 | 0.880 | 0.217 | 22.7 |
| para_200 | dense | 0.330 | 0.498 | 0.793 | 0.339 | 0.923 | 0.228 | 25.2 |
| para_200 | hybrid | 0.300 | 0.476 | 0.777 | 0.323 | 0.914 | 0.219 | 26.5 |
| para_200 | rerank | 0.244 | 0.425 | 0.748 | 0.294 | 0.898 | 0.203 | 425.9 |

### Pre-registered expectations vs test
1. Paragraph > fixed-token at equal size (gain ~1k): CONFIRMED in direction, 4/4 (+1.9, +3.3, +1.1, +2.3 points), smaller than on dev.
2. Hybrid > dense on Hit@1 with ~400-token chunks: NOT REPLICATED (token_400 0.362 vs 0.365; para_400 0.352 vs 0.357). Hybrid not better with ~200-token chunks: confirmed. Conclusion: hybrid does not reliably beat dense; the dev gain was likely noise.
3. Rerank < dense: CONFIRMED, 2/2 (Hit@1 0.310 vs 0.365; 0.244 vs 0.330), ~18x slower.
4. para_200 + dense highest gain at ~1k: CONFIRMED (0.339).
- Gains converge at ~2k again (0.185–0.239).
- Absolute scores are higher on test than dev; compare configurations only within a split.

## Phase 4b: paired bootstrap significance (test split, n = 1,308 questions, 10,000 resamples, seed 42)

| Comparison | Metric | Diff (pts) | 95% CI | p | Significant |
|---|---|---|---|---|---|
| para_400 vs token_400 (dense) | gain ~1k | +1.9 | [-0.5, +4.4] | 0.128 | no |
| para_400 vs token_400 (hybrid) | gain ~1k | +3.2 | [+0.7, +5.7] | 0.009 | yes |
| para_200 vs token_200 (dense) | gain ~1k | +1.1 | [-1.0, +3.3] | 0.321 | no |
| para_200 vs token_200 (hybrid) | gain ~1k | +2.2 | [+0.0, +4.5] | 0.048 | yes (borderline) |
| hybrid vs dense (token_400) | Hit@1 | -0.3 | [-2.8, +2.2] | 0.843 | no |
| hybrid vs dense (para_400) | Hit@1 | -0.5 | [-3.0, +2.1] | 0.742 | no |
| rerank vs dense (token_400) | Hit@1 | -5.5 | [-8.8, -2.2] | 0.002 | yes |
| rerank vs dense (para_200) | Hit@1 | -8.6 | [-11.6, -5.7] | <0.001 | yes |

- Reranker significantly worse than dense in both comparisons; survives Bonferroni correction (threshold ≈ 0.006). Most robust result.
- Hybrid vs dense: no significant difference; any true effect is within ~±3 points.
- Paragraph vs fixed-token: positive in all comparisons (4/4 test, 6/6 dev); significant with hybrid retrieval, not individually significant with dense alone. Does not survive Bonferroni correction; the claim rests on its consistency across comparisons.
- Pairing matters: per-question differences cancel question difficulty. Questions iterated in sorted order so bootstrap results are exactly reproducible.

## Phase 4c: figures (test split)

- `figures/gain_by_config.png`: gain over random at the ~1k-token budget for each chunker × retriever. Chunkers ordered fixed/paragraph at each size so each pair is adjacent. Rerank evaluated on two chunkers only (empty slots elsewhere). Bar heights are point estimates; significance is in the forest plot.
- `figures/significance.png`: forest plot of the 8 paired bootstrap comparisons from 4b; dot = mean difference (percentage points), bar = 95% interval, dashed line at zero.
- What they show: dense ≥ hybrid ≥ rerank within every chunker; paragraph > fixed at each size. Rerank intervals lie entirely below zero; hybrid-vs-dense intervals straddle zero; paragraph comparisons lie right of zero, clearing it only for hybrid.
- Generated by `make_figures.py` from `results/test/summary.csv` and `results/test/significance.csv` (written by `significance.py`), so they can be regenerated after any re-run.
- Possible polish for the write-up: colour non-significant comparisons grey in the forest plot.

## Phase 4d: failure analysis (test split)

- Hard questions (missed by all 10 test configurations at ~1k budget): 39 of 1,308 (3.0%). Most failures are configuration-specific, not universal.
- Sample: 20 of the 39 (seeded), reviewed against para_200 + dense retrievals (`results/test/failures.md`).

| Category | Count | Description |
|---|---|---|
| Questionable label | 6 | A retrieved chunk answers the question, but annotators marked a different passage (#1, 3, 10, 12, 17, 19). |
| Wording mismatch | 6 | Evidence uses different terms than the question, while wrong chunks share its words (e.g. "gold standard" vs "ground truth"; "regularisation" vs "dropout"). |
| Summary-section pull | 3 | Abstract/intro/conclusion restate the question's vocabulary and outrank the specific evidence. |
| Implicit answer | 3 | Yes/no or "does it include X" questions whose answer must be inferred; key terms never appear. |
| Answer in a table | 1 | The evidence text points to a table holding the actual answer. |
| Boilerplate pull | 1 | The acknowledgments section outranked the evidence. |

- ~30% of sampled hard failures are labelling artifacts: papers repeat facts and annotators mark one location, so the evaluation underestimates retrieval quality.
- Wording mismatch is the main genuine failure mode; neither dense nor BM25 handles it here (consistent with hybrid not helping).
- Summary-section pull is paper-specific. Possible future work: down-weight abstract/conclusion chunks for detail questions.
- Data quality: text extraction garbles numbers and LaTeX (#9, 10, 18); one heading-like evidence string passed the 5-word filter (#16).

## Phase 5a: FastAPI backend

- `api.py` exposes the pipeline over HTTP. Run with `uvicorn api:app --reload`; interactive docs at `http://localhost:8000/docs`.
- Uses **para_200 + dense**, the best configuration from the evaluation (top 5 × ~200 tokens ≈ the ~1k-token budget the results were measured at). `ask.py` still uses token_400_80; the API supersedes it for the demo.
- Endpoints:
  - `GET /health`: liveness check.
  - `GET /papers`: id and title of every validation-split paper.
  - `GET /papers/{paper_id}/questions`: that paper's QASPER questions (for suggestions in the UI); 404 if unknown.
  - `POST /ask` `{paper_id, question}`: returns `{answer, passages}`, where each passage has `number` (matches the [n] citations), `chunk_index`, `score` and `text`. 404 for an unknown paper, 422 for invalid input (Pydantic: question 3–500 chars), 503 if the database or Ollama is unreachable.
- Model, paper titles and questions load once at startup (`lifespan`); a fresh DB connection per request (fine for a demo; a pool would be used under load).
- Endpoints are plain `def`, not `async def`: the model, psycopg and Ollama calls are blocking, so FastAPI runs them in a thread pool.
- Added `dense_search` to `retrievers.py` (top-k with text and scores, same `(index, text, score)` shape `build_prompt` expects). `dense_rank` left unchanged so the evaluation stays reproducible.
- CORS allows `http://localhost:5173` for the React dev server (5b).
- Not deployed: needs Postgres data, the embedding model and a local LLM, which free hosting tiers can't run. Demo via screen recording instead.

### Full-circle check
- Paper 1912.01214, "which multilingual approaches do they compare with?": the Phase 1 failure case (token_400_80, evidence at rank 10, model refused).
- Through the API with para_200 + dense: evidence paragraph is passage 3; answer is correct ("pivoting, multilingual NMT (MNMT), and cross-lingual transfer without pretraining") and cites [3].
- Use this question for the demo recording.

## Phase 5b: React front end

- `frontend/`: React + Vite + Tailwind (JavaScript). Run with `npm run dev` from `frontend/` (http://localhost:5173), with the API running on port 8000. API URL set in `frontend/.env` (`VITE_API_URL`).
- Layout: left column picks a paper (text filter + dropdown over 281 validation papers) and lists its QASPER questions as one-click suggestions; right column has the question box, the answer and the five retrieved passages.
- The answer's `[n]` citations render as links to their passages; cited passages are highlighted. Each passage shows its similarity score and chunk index, so retrieval is visible, not just the LLM's output.
- Opens on the demo paper (1912.01214) by default.
- Components: `App` (state), `PaperPicker`, `AskBox` (⌘+Enter to ask; disabled under 3 characters or while loading), `Answer`, `Passages`. `api.js` wraps fetch; `citations.js` extracts cited numbers with a regex.
- Effect cleanup flag (`ignore`) prevents a slow question fetch for a previous paper from overwriting the current paper's list.
- Clear error states: API unreachable, database down, Ollama down.

### Gotcha found while testing
- With the Postgres container stopped, `/ask` raised an unhandled error. FastAPI's bare 500 response skips the CORS middleware, so the browser blocked it and the page showed "Can't reach the API" even though the API was running.
- Fix: catch `psycopg.OperationalError` around the DB call and raise `HTTPException(503, "Database unavailable…")`, which goes through the normal error path with CORS headers. Lesson: unhandled server errors can look like network failures in the browser.

### Test results
- Demo paper, "which multilingual approaches do they compare with?": correct answer citing [3]; passage 3 (chunk 32) highlighted as cited. Works end to end.
- TODO: "which datasets did they experiment with?" (Phase 1 generation failure): answered or refused with para_200?
- TODO: paper switching (1810.08699), Ollama-down and database-down messages, phone-width layout.


## Final conclusions (as of Phase 5b)

### Evaluation (test split, 1,308 questions)
1. **Cross-encoder reranker is significantly worse than plain dense retrieval** (−5.5 and −8.6 Hit@1 points on test), at ~18x the cost. Most robust result; survives Bonferroni correction.
2. **Hybrid (BM25 + dense) does not beat dense**: no significant difference on test; the dev-set gain did not replicate.
3. **Paragraph-aware chunking improved retrieval in every comparison** (4/4 test, 6/6 dev), significantly so with hybrid retrieval; with dense alone the effect is positive but not individually significant.
4. **Best configuration: para_200 + dense** (highest gain at the ~1k budget on dev and test).
5. **Chunking and retrieval choices matter most when the context budget is tight** (~1k tokens); at ~2k they converge.

### Failure analysis
6. **Only 3% of test questions fail in every configuration**, so most failures are configuration-specific. In a sample of 20, about 30% were labelling artifacts (a correct passage retrieved, but annotators marked another), so the evaluation underestimates retrieval quality. The main genuine failure mode is **wording mismatch** between question and evidence, which neither dense nor BM25 retrieval solved here.

### Applied
7. **The best configuration now powers the API and a web demo** (`POST /ask` + React front end, para_200 + dense, top 5). On the Phase 1 failure case (paper 1912.01214, "which multilingual approaches do they compare with?"), it retrieves the evidence as passage 3 and the model answers correctly with a citation, where the original token_400_80 setup ranked it 10th and the model refused. This is one illustrative example, not a measured result; the measured claims are points 1–6.