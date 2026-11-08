from loader import load_papers


def fixed_chunks(paragraphs, size = 200, overlap = 40):
    if(overlap >= size):
        raise ValueError("overlap must be smaller than size")


    text = " ".join(paragraphs)
    words = text.split()

    if not words:
        return []

    step = size - overlap
    
    chunks = []
    start = 0

    while True:
        chunk = " ".join(words[start:start + size])
        chunks.append(chunk)

        if(start + size >= len(words)):
            break
        else:
            start += step


    return chunks

def token_chunks(paragraphs, tokenizer, size=400, overlap=80):
    if overlap >= size:
        raise ValueError("Overlap must be smaller than size")

    text = " ".join(paragraphs)
    if not text.strip():
        return []

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)

    step = size - overlap
    chunks = []
    start = 0

    while True:
        end = min(start + size, n)
        char_start = offsets[start][0]
        char_end = offsets[end-1][1]
        chunks.append((text[char_start:char_end], char_start, char_end))

        if start + size >= n:
            break

        start += step

    return chunks

def para_chunks(paragraphs, tokenizer, max_tokens=400):
    text = " ".join(paragraphs)
    pos = 0

    chunks = []
    cur_start = None
    cur_end = None
    cur_tokens = 0
    
    for p in paragraphs:
        p_start = pos
        p_end = pos + len(p)
        pos = p_end + 1

        n = len(tokenizer(p, add_special_tokens=False)["input_ids"])

        if n> max_tokens:
            if cur_start is not None:
                chunks.append((text[cur_start:cur_end], cur_start, cur_end))
                cur_start = None
                cur_tokens = 0

            for t, s, e in token_chunks([p], tokenizer, max_tokens, max_tokens // 5):
                chunks.append((t, s + p_start, e + p_start))
            continue

        if cur_start is not None and  cur_tokens + n > max_tokens:
                chunks.append((text[cur_start: cur_end], cur_start, cur_end))
                cur_start = None
                cur_tokens = 0

        if cur_start is None:
            cur_start = p_start
        cur_end = p_end
        cur_tokens += n

    if cur_start is not None:
        chunks.append((text[cur_start:cur_end], cur_start, cur_end))

    return chunks



if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    sample = ["The quick brown fox jumps over the lazy dog."]
    for text, start, end in token_chunks(sample, model.tokenizer, size=4, overlap=1):
        print(repr(text), start, end)

    paper = load_papers(limit=1)[0]
    chunks = para_chunks(paper["paragraphs"],model.tokenizer, 400)

    print("Chunks:", len(chunks))
    print("Spans correct:", all(paper["text"][s:e] == t for t, s, e in chunks))
    print("Max tokens:", max(len(model.tokenizer(t, add_special_tokens=False)["input_ids"]) for t, s, e in chunks))