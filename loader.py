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

        resultList.append({"id": paper["id"], "title": paper["title"], "paragraphs": normalized, "text": " ".join(normalized)})

    return resultList


def load_questions(split = "validation", limit = None, min_words = 5):

    ds = load_dataset("allenai/qasper")
    data = ds[split]

    question_list = []
    papers_seen = 0

    for paper in data:

        if limit is not None and papers_seen >= limit:
            break
        papers_seen += 1

        qas = paper["qas"]

        for question, qid, answer_group in zip(qas["question"], qas["question_id"], qas["answers"]):
            evidence = set()

            for annotation in answer_group["answer"]:
                for ev in annotation["evidence"]:
                    if ev.startswith("FLOAT SELECTED"):
                        continue
                    norm = " ".join(ev.split())
                    if len(norm.split()) < min_words:
                        continue
                    evidence.add(norm)

            if evidence:
                question_list.append({
                    "paper_id": paper["id"],
                    "question_id": qid,
                    "question": question,
                    "evidence": list(evidence),
                })

    return question_list





if __name__ == "__main__":
    papers = load_papers(limit=3)
    for p in papers:
        print(p["id"], "|", p["title"])
        print("     paragraphs:", len(p["paragraphs"]))
        print("     first:", p["paragraphs"][0][:100])

    qs = load_questions();
    print("\nUsable questions:", len(qs))
    print("First:", qs[0]["question"])
    print("Evidence paragraphs:", len(qs[0]["evidence"]))