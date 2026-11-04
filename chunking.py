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


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    sample = ["The quick brown fox jumps over the lazy dog."]
    for text, start, end in token_chunks(sample, model.tokenizer, size=4, overlap=1):
        print(repr(text), start, end)
