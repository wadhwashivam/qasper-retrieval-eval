import numpy as np
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

def fetch_paper_chunks(conn, paper_id, chunker):
        return conn.execute(
            "SELECT chunk_index, char_start, char_end, content, embedding "
            "FROM chunks WHERE paper_id = %s AND chunker = %s ORDER BY chunk_index",
            (paper_id, chunker),
        ).fetchall()
    
def dense_rank(model, question, chunks):
    q_vec = model.encode(question, normalize_embeddings=True)
    doc_vecs = np.array([c[4].to_numpy() for c in chunks])
    scores = doc_vecs @ q_vec

    order = np.argsort(scores)[::-1]

    return [chunks[i][0] for i in order]

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def bm25_rank(question, chunks):
    corpus = [tokenize(c[3]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    scores = bm25.get_scores(tokenize(question))
    order = np.argsort(scores)[::-1]

    return [chunks[i][0] for i in order]

def rrf_fuse(rankings, k=60):
    scores = {}
    for ranking in rankings:         
        for pos, idx in enumerate(ranking, start = 1):
            scores[idx] = scores.get(idx, 0) + 1 / (k + pos)
    return sorted(scores, key = scores.get, reverse=True)

def hybrid_rank(model, question, chunks):
    return rrf_fuse([dense_rank(model,question,chunks), bm25_rank(question, chunks)])

def rerank_rank(reranker, question, chunks):

    pairs = [(question, c[3]) for c in chunks]

    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1]

    return [chunks[i][0] for i in order]




if __name__ == "__main__":
    print(rrf_fuse([[0,1,2],[2,0,1]]))
    print(tokenize("BIBREF19, MultiUN! The 2 datasets."))
