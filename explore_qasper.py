import json
from datasets import load_dataset

ds = load_dataset("allenai/qasper", revision="refs/convert/parquet")
print(ds)

paper = ds["validation"][0]
print(paper.keys())
print(paper["title"])

print("\n--- FULL TEXT ---")
ft = paper["full_text"]
print(type(ft), ft.keys())
print("Sections:", ft["section_name"])
print("Number of sections:", len(ft["paragraphs"]))
print("First section's paragraphs:", json.dumps(ft["paragraphs"][0], indent=2)[:1500])

print("\n--- QUESTIONS ---")
qas = paper["qas"]
print(qas.keys())
print("Number of questions:", len(qas["question"]))
print("First question:", qas["question"][0])
print("Its answers:", json.dumps(qas["answers"][0], indent=2)[:2500])