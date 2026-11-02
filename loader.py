from datasets import load_dataset

def load_papers(split = "validation", limit= None):

    ds = load_dataset("allenai/qasper")
    data = ds[split]

    resultList = []

    for paper in data:

        if limit is not None and len(resultList) >= limit:
            break
        
        sections = paper["full_text"]["paragraphs"]
        all_paragraphs = [p for section in sections for p in section]
        all_paragraphs = [paper["abstract"]] + all_paragraphs
        non_empty = [p for p in all_paragraphs if p.strip()]
        normalized = [" ".join(p.split()) for p in non_empty]

        resultList.append({"id": paper["id"], "title": paper["title"], "paragraphs": normalized})

    return resultList




if __name__ == "__main__":
    papers = load_papers(limit=3)
    for p in papers:
        print(p["id"], "|", p["title"])
        print("     paragraphs:", len(p["paragraphs"]))
        print("     first:", p["paragraphs"][0][:100])