from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

documents  = [
    "Cricket is a bat-and-ball game played between two teams of eleven players. "
    "A bowler delivers the ball toward the batter, who tries to hit it and score runs "
    "by running between the wickets. Test matches can last up to five days.",

    "Baseball is played between two teams of nine players over nine innings. "
    "The pitcher throws the ball to the batter, who tries to hit it and run around "
    "four bases. A run is scored when a player returns safely to home plate.",

    "Sourdough bread rises using a starter, a mix of flour and water that contains "
    "wild yeast and lactic acid bacteria. The bacteria give the bread its sour taste, "
    "and the slow fermentation can take twelve hours or more.",

    "A black hole forms when a massive star collapses under its own gravity. "
    "Its gravity is so strong that nothing, not even light, can escape once it "
    "crosses the boundary called the event horizon.",

    "Adults generally need seven to nine hours of sleep per night. During deep sleep "
    "the body repairs tissue, and during REM sleep the brain consolidates memories. "
    "Regular sleep deprivation is linked to weaker focus and immune function.",
]

doc_vecs = model.encode(documents, normalize_embeddings=True)
print("doc_vecs shape: ", doc_vecs.shape)

def search(question, k=2):
    q_vec = model.encode(question, normalize_embeddings=True)
    scores = doc_vecs @ q_vec

    print(f"\nQ: {question}")
    for i, s in enumerate(scores):
        print(f"   doc{i}: {s:.3f}")

    top_idx = np.argsort(scores)[::-1][:k]
    print(f"    Top {k}:")

    for i in top_idx:
        print(f"     {scores[i]:.3f}.  {documents[i][:60]}...")


search("Is cricket and baseball same sport?")
search("why am I forgetful after pulling an all-nighter?")
search("what is the capital of France? ")
