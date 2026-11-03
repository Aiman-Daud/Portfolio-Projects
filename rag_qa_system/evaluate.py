"""Runs every question in questions.txt through the RAG pipeline and saves results.csv.
Open results.csv, read each answer, and fill the 'correct' column with 1 or 0.
Then run:  python evaluate.py --score   to print your accuracy."""
import csv
import sys
import time

from rag import answer

if "--score" in sys.argv:
    rows = list(csv.DictReader(open("results.csv", encoding="utf-8")))
    marked = [r for r in rows if r["correct"].strip() in ("0", "1")]
    right = sum(int(r["correct"]) for r in marked)
    print(f"Accuracy: {right}/{len(marked)} = {right / len(marked):.0%}")
    sys.exit()

questions = [l.strip() for l in open("questions.txt", encoding="utf-8") if l.strip()]
with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question", "answer", "sources", "latency_s", "correct"])
    for q in questions:
        t = time.time()
        r = answer(q)
        w.writerow([q, r["answer"], r["sources"], round(time.time() - t, 2), ""])
        print("done:", q)
print("Saved results.csv. Fill the 'correct' column (1/0), then run: python evaluate.py --score")
