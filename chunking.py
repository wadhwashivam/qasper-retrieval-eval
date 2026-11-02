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


if __name__== "__main__":
    words = " ".join(f"w{i}" for i in range(12))

    for c in fixed_chunks([words], size=5, overlap=2):
        print(c)

    paper = load_papers(limit=1)[0]
    real_chunks = fixed_chunks(paper["paragraphs"])
    print("\nReal paper:", paper["title"])
    print("Chunks: ", len(real_chunks))
    print("Words in first chunk:", len(real_chunks[0].split()))
    print("Words in last chunk:", len(real_chunks[-1].split()))
