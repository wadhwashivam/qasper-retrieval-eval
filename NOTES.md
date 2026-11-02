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