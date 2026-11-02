from statistics import mean

from datasets import load_dataset

ds = load_dataset("allenai/qasper")
split = ds["validation"]

# --- counters ---
para_counts = []            # paragraphs per paper
empty_paragraphs = 0        # paragraph strings that are blank
total_questions = 0
total_annotations = 0       # one per annotator per question
unanswerable = 0
evidence_total = 0
float_evidence = 0          # evidence pointing at a table/figure
text_evidence = 0
text_evidence_verbatim = 0  # text evidence found exactly in full_text
questions_with_text_evidence = 0
mismatches = [];

for paper in split:
    # Level 1: the paper's body.
    # full_text["paragraphs"] is a list of sections, each a list of strings.
    sections = paper["full_text"]["paragraphs"]
    all_paragraphs = [p for section in sections for p in section]
    para_counts.append(len(all_paragraphs))
    empty_paragraphs += sum(1 for p in all_paragraphs if not p.strip())
    paragraph_set = set(all_paragraphs)  # for fast "is this evidence in the paper?" lookups

    # Level 2: the paper's questions (parallel lists, like full_text).
    qas = paper["qas"]
    total_questions += len(qas["question"])

    for answer_group in qas["answers"]:
        # Level 3: one entry per annotator for this question.
        has_text_evidence = False
        for annotation in answer_group["answer"]:
            total_annotations += 1
            if annotation["unanswerable"]:
                unanswerable += 1

            # Level 4: each evidence string this annotator cited.
            for ev in annotation["evidence"]:
                evidence_total += 1
                if ev.startswith("FLOAT SELECTED"):
                    float_evidence += 1
                else:
                    text_evidence += 1
                    has_text_evidence = True
                    if ev in paragraph_set:
                        text_evidence_verbatim += 1
                    else:
                        mismatches.append(ev)

        if has_text_evidence:
            questions_with_text_evidence += 1

# --- report ---
print(f"Papers (validation): {len(split)}")
print(f"Paragraphs per paper: avg {mean(para_counts):.1f}, "
      f"min {min(para_counts)}, max {max(para_counts)}")
print(f"Empty paragraph strings: {empty_paragraphs}")
print()
print(f"Questions: {total_questions}")
print(f"Annotator answers: {total_annotations} "
      f"({total_annotations / total_questions:.2f} per question)")
print(f"Unanswerable annotations: {unanswerable}")
print()
print(f"Evidence strings: {evidence_total}")
print(f"  Table/figure (FLOAT SELECTED): {float_evidence}")
print(f"  Text: {text_evidence}")
print(f"  Text found verbatim in full_text: {text_evidence_verbatim} "
      f"({text_evidence_verbatim / text_evidence:.1%})")
print()
print(f"Questions with at least one text evidence paragraph: "
      f"{questions_with_text_evidence} of {total_questions}")

print(f"\nMismatches found: {len(mismatches)}")
for ev in mismatches[:3]:
    print("---")
    print(ev[:300])