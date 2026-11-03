import json

with open("docs.json", encoding="utf-8") as f:
    docs = json.load(f)

big = []
for i in range(10):
    d = docs[i % len(docs)]           # cycle through the original documents
    big.append({"id": i + 1, "title": d["title"], "text": d["text"]})

with open("bench_docs.json", "w", encoding="utf-8") as f:
    json.dump(big, f, indent=2)