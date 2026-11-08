def evidence_spans(paper_text, evidence):
    spans = []
    for ev in evidence:
        pos = paper_text.find(ev)
        if pos != -1:
            spans.append((pos, pos + len(ev)))
    return spans


def coverage(chunk_start, chunk_end, ev_start, ev_end):
    overlap = max(0, min(chunk_end, ev_end) - max(chunk_start, ev_start))
    return overlap / (ev_end - ev_start)

def chunk_purity(chunk_start, chunk_end, ev_start, ev_end):
    overlap = max(0, min(chunk_end, ev_end) - max(chunk_start, ev_start))
    return overlap / (chunk_end - chunk_start)

def relevant_chunks(chunks, spans, threshold = 0.5):
    rel = set()
    for idx, cs, ce in chunks:
        for es, ee in spans:
            if coverage(cs, ce, es, ee) >= threshold or chunk_purity(cs, ce, es,ee) >= threshold:
                rel.add(idx)
                break

    return rel


if __name__ == "__main__":
    import os
    import psycopg
    from dotenv import load_dotenv
    from loader import load_papers, load_questions

    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"])

    paper = load_papers(limit=1)[0]
    q = load_questions(limit=1)[0]
    print("Q:", q["question"])

    spans = evidence_spans(paper["text"], q["evidence"])
    print(f"Evidence: {len(q['evidence'])}, found in text: {len(spans)}, spans: {spans}")

    chunks = conn.execute(
        "SELECT chunk_index, char_start, char_end FROM chunks "
        "WHERE paper_id = %s AND chunker = %s ORDER BY chunk_index",
        (paper["id"], "token_400_80"),
    ).fetchall()

    for idx, cs, ce in chunks:
        for es, ee in spans:
            c = coverage(cs, ce, es, ee)
            if c > 0:
                print(f"chunk {idx}: covers {c:.0%} of evidence")

    print("Relevant chunks:", relevant_chunks(chunks, spans))
    conn.close()