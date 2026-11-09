# RAG Project — Lab Notes

## Dataset: QASPER (validation split)

- 281 papers, 1,005 questions.
- Paragraphs per paper: avg 48.4, min 7, max 292. Papers are long enough that chunking matters.
- 328 empty paragraph strings across the split. The loader skips them.
- 1.76 annotator answers per question on average. Rule for Phase 2: a paragraph is relevant if *any* annotator cited it.
- 163 annotator answers marked unanswerable.
- Evidence strings: 2,808 total. 253 point at tables/figures ("FLOAT SELECTED"), 2,555 are text.
- 91.9% of text evidence (2,349 of 2,555) appears verbatim in `full_text`.
  - Mismatch causes seen: section headings cited as evidence (e.g. "Languages", "Datasets"), and spacing differences (e.g. "BIBREF20 , which").
  - Phase 2 plan: match by overlap after whitespace normalization; drop evidence shorter than a few words.
- 892 of 1,005 questions have at least one text evidence paragraph. The other 113 are excluded from retrieval evaluation (no text evidence).
- After filtering (text evidence only, ≥5 words, deduplicated across annotators): **889 usable questions** on the dev set (validation split). 3 questions were dropped because their only evidence was heading-length.

## Design decisions

- Retrieval is scoped to the question's own paper (`WHERE paper_id = ...`). QASPER questions refer to "they"/"this paper", so cross-paper search would mix chunking quality with paper identification.
- Loader: abstract first, then all `full_text` paragraphs in order, empty ones removed, whitespace normalized with `" ".join(p.split())`.
- Embeddings: `BAAI/bge-small-en-v1.5`, 384 dims, normalized, so cosine similarity = dot product.
- Exact vector search (no HNSW index) so approximation noise doesn't affect results.
- Each chunk stores `char_start`/`char_end` positions in the paper's joined text (`" ".join(paragraphs)`). Evidence relevance is computed by span overlap: a chunk is relevant if it covers ≥50% of an evidence paragraph's span.
- Relevance rule validated on paper 1912.01214, Q "which multilingual approaches do they compare with?": evidence span (19730, 20675); chunk 13 covers 100%, chunk 14 covers 10% (overlap carry-over). Rule output {13} matches the manual diagnosis from Phase 1.

## Finding: word-based chunks exceeded the model's token limit

- First chunker: fixed 200 words, 40-word overlap.
- On the first 5 papers (119 chunks), the longest chunk was **655 tokens**; bge truncates input at 512 tokens.
- Chunks over 512 tokens: [4] of 119.
- Cause: scientific text tokenizes heavily (LaTeX, BIBREF/TABREF placeholders, numbers) — over 3 tokens per word in dense passages.
- Consequence: the tail of long chunks would be silently dropped from embeddings, biasing retrieval results.
- Fix: switched to token-based chunking using the model's own tokenizer (sizes in model tokens), slicing the original text by token character offsets so casing and spacing are preserved.

Python dense ranking reproduces the SQL results exactly (0.287 / 0.597 / 0.759 / 0.490).

## Index runs

| Chunker | Papers | Chunks | Seconds | Max tokens |
|---|---|---|---|---|
| fixed_200_40 (words) | 5 | 119 | 1.4 | 655 |
| token_400_80 | 5 | 92 | 1.1 | 403 | 
| token_400_80 | 281 | 4619 | 27.3 | 404


## Phase 1: end-to-end pipeline

- Pipeline: `loader.py` → `chunking.py` → `index.py` (indexing time) and `ask.py` (query time).
- Generation: `llama3.1:8b` via Ollama, temperature 0 (deterministic, reproducible), top k = 5.
- Ollama default context window is 4,096 tokens and truncates silently. 5 chunks × ~400 tokens + instructions ≈ 2,500 tokens, so it fits. Raising k to 10 would exceed it.
- Prompt tells the model to answer only from the numbered passages, cite them as [n], and say "The provided context does not answer this question." if the answer is absent.

## Phase 2 results: baseline (dev set, validation split)

| Config | Questions | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| token_400_80 + dense | 888 | 0.287 | 0.597 | 0.759 | 0.490 |
| token_400_80 + random | 888 | 0.131 | 0.349 | 0.525 | 0.314 |

- 889 usable questions; 1 skipped (evidence not locatable), 0 skipped for no covering chunk. Eval runs in ~14 s.
- Dense clearly beats random, but random Hit@5 is already 0.525 (~16 chunks per paper, so top 5 ≈ a third of the paper). Hit@1 and MRR are the more discriminating metrics.
- Dense ranks the gold evidence first only 29% of the time, so there is real headroom for Phase 3.
- Methodology issue for Phase 3: chunkers with different chunk counts have different random baselines. Compare at equal token budget, not equal k.
- Deterministic: two runs give identical numbers.

## Phase 3a: chunk size (dense retrieval, symmetric relevance rule, 888 questions)

| Chunker | Hit@1 | Hit@5 | MRR | Budget ~1k | Random ~1k | Budget ~2k | Random ~2k |
|---|---|---|---|---|---|---|---|
| token_400_80 | 0.287 | 0.761 | 0.490 | 0.616 | 0.364 | 0.809 | 0.592 |
| token_200_40 | 0.293 | 0.650 | 0.459 | 0.686 | 0.382 | 0.858 | 0.610 |

- Budget metric validated: random scores are nearly equal across chunk sizes, unlike Hit@5 (random 0.527 vs 0.339).
- 200-token chunks beat 400-token chunks at equal text budget: +5.2 points over random at ~1k, +3.1 at ~2k.
- Hit@k and MRR are not comparable across chunk sizes; use them only to compare retrievers on the same chunker.
- To do in Phase 4: paired significance test on per-question results.

## Phase 3b: paragraph-aware chunking (dense retrieval, 888 questions)

| Chunker | Chunks | Hit@1 | MRR | Budget ~1k | Random ~1k | Gain ~1k | Budget ~2k | Random ~2k | Gain ~2k |
|---|---|---|---|---|---|---|---|---|---|
| token_400_80 | 4,619 | 0.287 | 0.490 | 0.616 | 0.364 | 0.252 | 0.809 | 0.592 | 0.217 |
| para_400 | 4,636 | 0.310 | 0.495 | 0.664 | 0.382 | 0.282 | 0.848 | 0.626 | 0.222 |
| token_200_40 | — | 0.293 | 0.459 | 0.686 | 0.382 | 0.304 | 0.858 | 0.610 | 0.248 |
| para_200 | 10,413 | 0.302 | 0.469 | 0.771 | 0.416 | 0.355 | 0.900 | 0.655 | 0.245 |

Gain = dense minus random at the same budget (the fair comparison across chunkers).

- Paragraph-aware chunking beats fixed-token chunking at a ~1k-token budget (gain over random: +3.0 points at 400, +5.1 at 200), and raises Hit@1 (0.310 vs 0.287 at 400).
- At ~2k tokens the advantage disappears (gains within ~0.5 points): chunk boundaries matter most when the context budget is tight.
- Best so far: para_200, 0.771 at ~1k tokens (gain 0.355 over random).
- Variable chunk lengths make random baselines differ across chunkers, so compare by gain over random.
- Hit@1 and MRR across chunkers are shown for reference only; they aren't comparable across chunk sizes.
- To do in Phase 4: paired significance test on per-question results.
- Re-test with paragraph-aware chunking: para_200 ranks the evidence **3rd** (vs 10th with token_400_80); found within ~1k tokens. para_400 rank: [fill in].

## Phase 3c: BM25 and hybrid (RRF, k=60), 888 questions

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

- BM25 index built per question over that paper's chunks only (IDF is within-paper).
- On 400-token chunks, hybrid beats dense and BM25 on every metric (Hit@1 0.287 → 0.323): the retrievers are complementary.
- On para_200, hybrid is worse than dense (Hit@1 0.302 → 0.280). BM25 is much weaker on short chunks (12-point Hit@1 gap vs 4.5 on 400-token chunks), and equal-weight RRF lets the weaker retriever drag the stronger one down.
- Gains don't stack: paragraph chunking + dense beats paragraph chunking + hybrid. Hybrid helps only when the two retrievers are comparably strong.
- Best at the ~1k budget so far: para_200 + dense (0.771).
- Weighted RRF is possible but would mean tuning on dev; skipped to avoid overfitting.

## First test: paper 1912.01214 (4 QASPER questions)

| Question | Result |
|---|---|
| which multilingual approaches do they compare with? | Refused. Retrieval failure (see below) |
| what are the pivot-based baselines? | Answered, cited [3] (chunk 3) | Refused. Generation failure: Europarl/MultiUN appear in chunks 10, 11, 16, all in the top 5 (chunk 11 ranked first), but the model still refused. Likely cause: strict refusal instruction makes the 8B model over-cautious when passages don't match the question's wording.
| which datasets did they experiment with? | Refused. [Europarl check: retrieval or generation failure] |
| what language pairs are explored? | Refused. Not yet diagnosed |

### Failure case 1: evidence split across chunks

- Question: "which multilingual approaches do they compare with?"
- Gold evidence is in chunk 13 (found with SQL `LIKE`), which ranked **10th of 19**, similarity 0.689.
- Cause: chunk 13 opens with training setup (masking strategy, BPE), and the evidence paragraph sits at its tail. The paragraph's last sentence spills into chunk 14 (rank 6) through the overlap. The fixed-size chunker split the evidence and mixed it with unrelated text, diluting the chunk's embedding.
- Hypothesis for Phase 3: a paragraph- or section-aware chunker keeps the evidence paragraph whole and should rank it higher.

### Observations

- **Flat scores:** all 19 chunks scored between 0.637 and 0.726 for this question. When a question uses the paper's own topic vocabulary ("multilingual"), dense retrieval can't separate the chunks.
- **Random baseline needed:** the paper has only 19 chunks, so top 5 covers ~26% of it. A random retriever would hit the evidence about a quarter of the time. Recall@k must be reported alongside a random baseline.
- **Two failure types:**
  - Retrieval failure: evidence not in the top k, so refusing is correct.
  - Generation failure: evidence in the top k, but the model refuses or misses it.
- Phase 2 evaluates retrieval alone, with no LLM, so chunker and retriever quality are measured without generation noise mixed in.
- **Strict refusal tradeoff:** the "say exactly" refusal rule prevents made-up answers but causes wrongful refusals with an 8B model. Candidate side experiment: softer wording or a larger model (Gemma 4).