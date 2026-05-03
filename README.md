# Movie Q&A with Retrieval-Augmented Generation

A local-only RAG pipeline that answers natural-language questions about
~41,000 movies. Built end-to-end on Colab with no hosted API calls.

**Stack:** BGE bi-encoder → FAISS index → MS-MARCO cross-encoder re-ranker
→ Qwen 2.5-3B-Instruct (4-bit) generator. Evaluated with custom retrieval
metrics, RAGAS, and a custom faithfulness implementation.

## Results

On a 20-query test set (standard, adversarial, sequel-confusion, unanswerable):

| Metric | Bi-encoder only | + Cross-encoder |
|---|---|---|
| Recall@5 | 0.625 | **0.750** |
| MRR | 0.451 | **0.484** |

Strict prompt cuts hallucinations on unanswerable queries from 3/4 → 1/4.
Median end-to-end query latency: ~38 ms.

## Quick start

1. Open `CS455HW2_merged.ipynb` in Colab (GPU runtime).
2. Place `movies_metadata.csv` and `credits.csv` from
   [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
   in `Drive/MyDrive/CS455HW2/Data/`.
3. Run all cells. The graded entrypoint is `run_query(query: str) -> dict`.

Runtime: ~50 min on T4, ~5 min on A100, plus ~30 min for RAGAS.

## What's in the notebook

- Three controlled ablations (bi-encoder size, top-k sweep, prompt strictness)
- Two before/after fixes (title boosting for sequel-confusion;
  rerank-confidence threshold for fictional-title hallucinations)
- A three-layer ensemble extension that catches semantic-trap cases via
  year-mismatch metadata
- A custom faithfulness metric that survives the limits of small local
  judge LLMs (RAGAS Faithfulness collapses to 0.0 with a 3B judge —
  diagnosed as a JSON-schema mismatch, not a metric failure)
- Failure diary with 6 cases including a prompt-brittleness finding
  (`"Who directed Barbie?"` and `"Who directed Barbie"` produce different
  answers under greedy decoding)

Full discussion in the Analysis Report at the end of the notebook.

## Files

- `CS455HW2_merged.ipynb` — main notebook
- `hw2_promptlog.md` — log of AI-assisted work (per assignment Section 9)

## Context

CS 455/555 — *Large Language Models*, Sabancı University. The "no hosted
APIs, free-tier Colab T4" constraint is what makes the engineering
interesting: most production RAG tutorials assume GPT-4 as both generator
and judge.
