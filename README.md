# Movie Q&A with Retrieval-Augmented Generation

A local-only RAG (Retrieval-Augmented Generation) system that answers
natural-language questions about ~41,000 movies. Built end-to-end on
Colab with no hosted API calls: BGE bi-encoder embeddings, FAISS
indexing, MS-MARCO cross-encoder re-ranking, and a 4-bit-quantized
Qwen 2.5-3B-Instruct generator. Evaluated with custom retrieval
metrics, RAGAS, and a custom faithfulness implementation that survives
the limits of small local judge LLMs.

> **Course context:** This was built as Homework 2 for CS 455 / 555
> (*Large Language Models*) at Sabancı University. The constraint of
> "no hosted APIs, free-tier Colab T4 only" is what makes the
> engineering interesting — most production RAG tutorials assume
> GPT-4 or Claude as both the generator and the judge.

---

## What this system does

Given a question like *"Who directed Inception?"* or *"A movie about a
boxing underdog"*, the pipeline:

1. Embeds the query with `BAAI/bge-small-en-v1.5` (384-dim).
2. Retrieves the 20 nearest documents from a FAISS index built over
   41,367 unique movie descriptions.
3. Re-ranks those 20 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`
   to a top-5.
4. Generates a grounded answer with `Qwen/Qwen2.5-3B-Instruct`
   (4-bit nf4 quantization), using a strict prompt that requires
   answering only from retrieved context and refusing otherwise.

The graded entrypoint is `run_query(query: str) -> dict`, which
returns `{"answer": str, "retrieved_titles": list[str]}` for any
input string (including empty, non-English, or nonsense queries) and
is guaranteed not to raise.

---

## Headline results

On a 20-query test set (5 standard, 7 adversarial-paraphrase, 4
sequel-confusion, 4 unanswerable):

| Metric                              | Value    |
|-------------------------------------|----------|
| Recall@5 (bi-encoder + cross-encoder) | **0.750** |
| Recall@5 (bi-encoder only)          | 0.625    |
| MRR (after re-rank)                 | **0.484** |
| MRR (bi-encoder only)               | 0.451    |
| Hallucinations on unanswerable queries (strict prompt) | **1 / 4** |
| Hallucinations on unanswerable queries (permissive prompt) | 3 / 4 |
| End-to-end query latency (median)   | ~38 ms   |

The cross-encoder lifts Recall@5 by 12.5 absolute points; the strict
prompt cuts hallucinations on unanswerable queries by 67%. See the
[Analysis Report](#analysis-report) section of the notebook for full
discussion.

---

## Architecture
