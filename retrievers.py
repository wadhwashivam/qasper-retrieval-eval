import numpy as np

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