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

## Design decisions

- Retrieval is scoped to the question's own paper (`WHERE paper_id = ...`). QASPER questions refer to "they"/"this paper", so cross-paper search would mix chunking quality with paper identification.
- Loader: abstract first, then all `full_text` paragraphs in order, empty ones removed, whitespace normalized with `" ".join(p.split())`.
- Embeddings: `BAAI/bge-small-en-v1.5`, 384 dims, normalized, so cosine similarity = dot product.
- Exact vector search (no HNSW index) so approximation noise doesn't affect results.
- Each chunk stores `char_start`/`char_end` positions in the paper's joined text (`" ".join(paragraphs)`). Evidence relevance is computed by span overlap: a chunk is relevant if it covers ≥50% of an evidence paragraph's span.

## Finding: word-based chunks exceeded the model's token limit

- First chunker: fixed 200 words, 40-word overlap.
- On the first 5 papers (119 chunks), the longest chunk was **655 tokens**; bge truncates input at 512 tokens.
- Chunks over 512 tokens: [4] of 119.
- Cause: scientific text tokenizes heavily (LaTeX, BIBREF/TABREF placeholders, numbers) — over 3 tokens per word in dense passages.
- Consequence: the tail of long chunks would be silently dropped from embeddings, biasing retrieval results.
- Fix: switched to token-based chunking using the model's own tokenizer (sizes in model tokens), slicing the original text by token character offsets so casing and spacing are preserved.

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