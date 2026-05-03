# HW2 Prompt Log

**Course:** CS 455 / 555 — Large Language Models
**Assignment:** HW2, Movie Q&A with RAG
**Tool used throughout:** Claude (Anthropic)

This log records the non-trivial AI interactions that shaped the
submitted notebook. Trivial autocomplete-style queries and one-off
syntax lookups are not included. Each entry follows the format from
Section 9 of the assignment spec.

---

## Entry 1 — Initial review of draft notebook against the spec

**Task ID:** Whole-notebook audit (Sections 1–8 of the spec).

**Tool:** Claude.

**Prompt:** "I uploaded my draft notebook and the assignment .docx.
Evaluate the notebook, point out missing or wrong parts, and show me
how it should be done. Also explain the RAGAS slowness — that cell ran
for hours and never finished."

**Output summary:** Produced a prioritised list of gaps ranked by
graded point impact. Identified that `run_query()` was missing entirely
(25 pts at risk), the Failure Diary section did not exist (10 pts),
RAGAS only computed Faithfulness instead of all three required metrics,
and ablation interpretation paragraphs were placeholder text.
Diagnosed the RAGAS cell hang as `max_new_tokens=96` cutting JSON
mid-token and `timeout=300` being too short for a sequential 4-bit Qwen
3B judge. Proposed a fix: raise `max_new_tokens` to 512, raise timeout
to 900 s, add `answer_relevancy`, and pass local embeddings.

**Verified/changed:** Cross-checked every claim against the spec
directly. Implemented `run_query()` with the contract from Section 7
(returns dict, never raises, exact-string titles). Built the Failure
Diary as a separate section. Applied the RAGAS config changes — cell
went from "hangs forever" to "completes in ~47 minutes." Did not
accept the suggestion to add `context_precision`; it requires a
`reference` column the test set does not have, which I caught when the
cell raised a `ValueError`.

---

## Entry 2 — Diagnosing the `context_precision` ValueError

**Task ID:** Task 5.3 / Section 11 (RAGAS evaluation).

**Tool:** Claude.

**Prompt:** "After `make_ragas_dataset` it says: `ValueError: The
metric [context_precision] that is used requires the following
additional columns ['reference'] to be present in the dataset.` Why?"

**Output summary:** Explained that `context_precision` is a
*reference-based* metric — it needs ground-truth answers. Two options
offered: (1) write synthetic gold answers (rejected because
unanswerable queries have none and writing answers for the rest would
be circular), or (2) drop `context_precision` from RAGAS and implement
Context Relevance manually with a HIGH/MEDIUM/LOW prompt mapped to
{1.0, 0.5, 0.0}.

**Verified/changed:** Took option 2. Removed `context_precision` from
the RAGAS metrics list. Added Section 12's custom Context Relevance
implementation. Verified the custom metric ran cleanly on all 20 test
examples (whereas RAGAS Faithfulness still failed on most). The
cleaner architecture was a side benefit: the custom implementation
bypasses RAGAS's pydantic parser entirely.

---

## Entry 3 — Why RAGAS Faithfulness collapsed to 0.0000

**Task ID:** Task 5.3 / Section 11 (RAGAS evaluation).

**Tool:** Claude.

**Prompt:** "RAGAS finished but reported `faithfulness: 0.0` and
`answer_relevancy: 0.66` for permissive, `0.0 / 0.50` for strict. The
log is full of `fix_output_format failed to parse output` and
`statement_generator_prompt failed to parse output`. Explain these
errors."

**Output summary:** Identified the dominant failure pattern: Qwen
2.5-3B emits valid JSON in a slightly different shape than RAGAS's
pydantic schema requires — `{"statements": [{"text": "..."}]}` instead
of `{"statements": ["..."]}`. The schema expects a list of strings;
Qwen emits a list of objects. RAGAS's `fix_output_format` retry uses
the same model, so it fails identically. With `raise_exceptions=False`,
RAGAS substitutes 0 for failed jobs, which is why the average collapsed
to exactly 0.0000. Conclusion: model-capacity problem, not metric
problem — RAGAS Faithfulness works as designed with a 7B+ judge.

**Verified/changed:** Inspected several actual Qwen JSON outputs from
the error log to confirm the diagnosis matched the parser errors.
Wrote up the explanation in Section 2.2 of the Analysis Report. Built
the Custom Faithfulness fallback in Section 12 using minimal output
shapes (numbered lists, single-word YES/NO) that small judges handle
reliably. Confirmed on a re-run that strict-prompt RAGAS Faithfulness
is now 0.25 (not 0.00), which validated both the diagnosis and the
secondary claim that prompt strictness improves evaluation
tractability.

---

## Entry 4 — Implementing the title-boosting fix

**Task ID:** Section 6 (Failure Diary), Fix #1.

**Tool:** Claude.

**Prompt:** "Sequel-confusion fails on 'Who directed Alien?' — the
1979 original sits at rank 4 behind Alien 2: On Earth and The Alien
Saga. Suggest a fix and show me how to implement it with before/after
evidence."

**Output summary:** Hypothesis was that the title appears once in
`document_text` (once as `Title: Alien`) but sequels' overviews are
longer and more keyword-dense, so cosine similarity favours sequels.
Proposed prepending `Movie title: <title>. <title> (<year>).` to the
embedded text, building a second FAISS index from boosted documents,
and re-running the *Alien* probe. Two more probe queries were
suggested for the demonstration: *"Who directed The Godfather?"* and
*"Who directed The Terminator?"*

**Verified/changed:** Implemented the fix as Section 14. Ran on all
suggested probes plus a fourth (*Star Wars*). Results: *Alien* moved
rank 4 → 1, *Godfather* moved 3 → 1, *Star Wars* moved 2 → 1,
*Terminator* stayed at 1 (already correct, kept as a no-op control).
Retained the *Star Wars* probe specifically because the suggestion
flagged it as a likely failure case (the *Empire of Dreams*
documentary outranking the original) but on my actual run the boost
succeeded — worth noting in the report.

---

## Entry 5 — Designing the rerank-confidence threshold and the ensemble

**Task ID:** Section 6 (Failure Diary), Fix #2 and the Section 7.4
ensemble extension.

**Tool:** Claude.

**Prompt:** "The other failure mode is hallucination on unanswerable
queries (Oppenheimer, fictional titles). Design a fix that uses the
cross-encoder's score as a confidence signal and refuses below a
threshold. Then suggest two more probes for it."

**Output summary:** Implemented the single-threshold fix (refuse when
top-1 rerank score < 0.0). Suggested *"Quantum Banana Republic"*
(another purely fictional title, to confirm generalisation) and *"Who
directed Barbie?"* (post-2017 semantic-trap probe). Predicted that
fictional probes would refuse and Barbie would slip through because
the corpus contains old Barbie animated films that score high. Later,
when I asked how to handle the residual semantic-trap, suggested a
three-layer ensemble: stricter threshold, year-mismatch metadata
check, and a stricter system prompt.

**Verified/changed:** Implemented Fix #2 as Section 15 and the
ensemble as Section 7.4 / Fix #3. Set ensemble's threshold to 1.5
(higher than 0.0). Verified on five probes:
- *Galactic Dishwasher* → Layer 1 fired (rerank −5.88), refused. ✓
- *Quantum Banana Republic* → similar pattern. ✓
- *Oppenheimer 2023* → Layer 2 fired (year 2023 vs doc 1981), refused. ✓
- *Inception* → all three layers passed, LLM answered correctly. ✓
- *Barbie* → all layers passed (rerank 6.96 is high), LLM answered
  with old Barbie films — exactly the predicted slip-through.

The Barbie slip-through is honestly documented as the residual
failure mode of the fix.

---

## Entry 6 — Interpreting the unexpected Ablation B pattern

**Task ID:** Task 5.2 (top-k sweep, Ablation B).

**Tool:** Claude.

**Prompt:** "The expected behaviour for Ablation B is that the gain
from k=10 to k=20 should be smaller than from k=5 to k=10. Mine is the
opposite — both jumps are roughly equal in R@5 and the second jump is
*bigger* in MRR. Why?"

**Output summary:** Explained that the textbook expectation assumes
gold documents typically live at ranks 1–7 in the bi-encoder's
output. My test set is heavy in adversarial paraphrases and
sequel-confusion items where gold gets buried at deeper ranks because
distractors (sequels, franchise documentaries, keyword-rich
narratively-similar films) outrank the original. Concrete examples:
*Cardboard Boxer*, *Raging Bull*, *Gladiator 1992* outrank *Rocky*
for "boxing underdog"; *The Alien Saga* and *Alien 2: On Earth*
outrank the 1979 *Alien*. Also explained why MRR grows faster than
R@5 in the second jump: newly-visible candidates that the
cross-encoder rescues tend to land at rank 1 (full MRR contribution)
rather than just somewhere in the top 5.

**Verified/changed:** Cross-referenced the explanation against my
`inspect_topk_failures(retrieve_k=5)` output — the failures listed
there (Inception, Jaws, Godfather, Titanic, Rocky, Alien) match the
adversarial+sequel-confusion categories the explanation predicts.
Wrote up the explanation in Section 3.2 of the Analysis Report and
added a "what gold means" subsection because the term was used
without definition in earlier drafts.

---

## Entry 7 — Diagnosing the prompt-brittleness finding

**Task ID:** Section 4.6 of the Analysis Report (accidental finding).

**Tool:** Claude.

**Prompt:** "Look at this: `Who directed Barbie?` and `Who directed
Barbie` (no question mark) produce different answers — different
directors attributed to different films. Same retrieval, same
greedy decoding. What's going on?"

**Output summary:** Distinguished two distinct mechanisms: (1)
tokenisation propagation — the bi-encoder produces a slightly
different query vector when the input differs by one token, and the
cross-encoder's top-1 score actually shifted from 6.96 to 7.36
between the two runs. (2) Conditioning sensitivity — even with
identical retrieved context, the generator's prompt now ends with a
different final token, which shifts the next-token distribution and
can propagate through the entire generation. Greedy decoding
preserves per-prompt determinism but does not protect against
surface perturbations.

**Verified/changed:** Confirmed by re-running each query
individually — outputs are reproducible per input but diverge
between inputs. Wrote up the case as Section 4.6 of the Analysis
Report, explicitly documenting that it slips through every metric in
the evaluation suite (RAGAS Faithfulness, answer relevancy,
hallucination count) because both answers are individually
"reasonable." Used the Lessons Learned section to argue this implies
a need for input normalisation or a larger generator.




# HW2 Prompt Log — Movie Q&A with RAG

## Entry 8

**Task ID:** Section 1 / Assignment Understanding  
**Tool:** ChatGPT  
**Prompt:** “We will be studying this homework in this chat.”  
**Output summary:** ChatGPT summarized the assignment as a Movie Q&A Retrieval-Augmented Generation pipeline involving dataset preprocessing, bi-encoder retrieval, vector database indexing, cross-encoder re-ranking, local LLM generation, ablations, evaluation, RAGAS, and prompt logging.  
**Verified/changed:** I used this summary only as a roadmap. I checked the homework PDF/Word document myself and followed the required section order from the assignment.

---

## Entry 9

**Task ID:** Section 4 / Evaluation Concepts  
**Tool:** ChatGPT  
**Prompt:** “Explain what ablations, retrieval metrics and RAGAS is.”  
**Output summary:** ChatGPT explained that ablations compare two controlled configurations while keeping other variables constant, retrieval metrics measure whether correct movie documents are retrieved, and RAGAS evaluates answer faithfulness, context relevance, and answer relevance.  
**Verified/changed:** I used the explanation to understand why the assignment requires both custom retrieval metrics and RAGAS. I did not directly copy code from this interaction.

---

## Entry 10

**Task ID:** Section 2 / Dataset and Subsampling  
**Tool:** ChatGPT  
**Prompt:** “Explain why we shouldn't subsample permanently based on the paper.”  
**Output summary:** ChatGPT explained that permanent random subsampling can remove original movies while keeping sequels, causing franchise queries such as “Who directed Toy Story?” to retrieve only a sequel. It also explained that this can create sample-bias failures that may look grounded but still be wrong.  
**Verified/changed:** I used this explanation in my notes to justify why the final corpus should use the full preprocessed dataset. I kept subsampling only as a possible development shortcut, not for final indexing or evaluation.

---

## Entry 11

**Task ID:** Task 1 / Inspecting `movies_metadata.csv` and `credits.csv`  
**Tool:** ChatGPT  
**Prompt:** “Explain me the content of movies metadata and credits csvs. Provide me a python code for both to exemplify their content such as `.head`.”  
**Output summary:** ChatGPT described the key columns in `movies_metadata.csv` such as `id`, `title`, `overview`, `release_date`, and `genres`, and the key columns in `credits.csv` such as `id`, `cast`, and `crew`. It provided inspection code using `.head()`, `.columns.tolist()`, `.info()`, and missing value checks.  
**Verified/changed:** I ran the inspection cells in Colab and confirmed the file shapes: `credits.csv` had 45,476 rows and `movies_metadata.csv` had 45,466 rows. I used the output to decide which columns were needed for the final `document_text`.

---

## Entry 12

**Task ID:** Task 1 / Understanding Cast and Crew IDs  
**Tool:** ChatGPT  
**Prompt:** “What cast_ID and crew_ID represent there are same ids with different names?”  
**Output summary:** ChatGPT explained the difference between movie-level `id`, person-level `id`, `cast_id`, and `credit_id`. It clarified that the movie-level `id` is used for merging the two CSV files, while `cast_id` and `credit_id` should not be used as global person identifiers.  
**Verified/changed:** I used only the movie-level `id` to merge `movies_metadata.csv` and `credits.csv`. For people information, I extracted director names from `crew` and top cast names from `cast`, rather than relying on `cast_id` or `credit_id`.

---

## Entry 13

**Task ID:** Task 1 / Data Preprocessing  
**Tool:** ChatGPT  
**Prompt:** I pasted the assignment preprocessing requirements: drop missing titles, drop missing or empty overviews, drop duplicate titles while keeping the longer overview, use the full corpus, and construct `document_text`.  
**Output summary:** ChatGPT produced preprocessing code that parses JSON-like columns, extracts genres, cast names, and director, converts IDs safely, drops malformed rows, removes duplicate titles by longer overview, and constructs a `document_text` field combining title, year, genres, director, cast, and overview.  
**Verified/changed:** I ran the preprocessing code in Colab. When Pandas showed a `SettingWithCopyWarning`, I asked about it and added `.copy()` after filtering operations. I verified the final corpus had 41,367 rows, no missing titles, no missing overviews, no empty overviews, no duplicate titles, and no missing `document_text`.

---

## Entry 14

**Task ID:** Task 1 / Reporting Corpus Statistics  
**Tool:** ChatGPT  
**Prompt:** “I need these (mean/median/max), and 3 example `document_text` strings from your own corpus.”  
**Output summary:** ChatGPT gave code to compute whitespace-based overview token counts and print mean, median, and max values, plus random examples of `document_text` using `sample(3, random_state=42)`.  
**Verified/changed:** I ran the code and reported: final corpus size = 41,367, mean overview length = 56.386 tokens, median = 50.0, max = 187. I used random examples instead of the first rows because the alphabetically first titles began with symbols such as `#` and `$`.

---

## Entry 15

**Task ID:** Task 2 / Embedding and FAISS Indexing  
**Tool:** ChatGPT  
**Prompt:** “Now that I have the merged `document_text` which is `corpus_df`, I need to feed them into a bi-encoder and before that I need to convert them to vectors, meaning encoding them. What is the next step?”  
**Output summary:** ChatGPT clarified that I do not manually convert text before the bi-encoder; instead, the bi-encoder receives text and outputs embeddings. It provided code for loading `BAAI/bge-small-en-v1.5`, encoding all `document_text` values, normalizing embeddings, building a FAISS `IndexFlatIP`, and implementing `retrieve_faiss()`.  
**Verified/changed:** I ran the embedding step on GPU. The embedding shape was `(41367, 384)`, encoding time was about 130.45 seconds, throughput was about 317.11 docs/sec, and embedding memory was about 0.059 GB. I verified that FAISS contained 41,367 vectors.

---

## Entry 16

**Task ID:** Task 2 / Debugging FAISS Invalid Results  
**Tool:** ChatGPT  
**Prompt:** I showed a screenshot where every FAISS retrieval result was the same Japanese title and the scores were extremely negative, like `-340282346638528859...`.  
**Output summary:** ChatGPT explained that this value is close to the minimum float32 value and indicates invalid FAISS results, commonly caused by an empty index or invalid indices such as `-1`. It also explained that `corpus_df.iloc[-1]` would select the last row, which explained why the same Japanese title appeared repeatedly.  
**Verified/changed:** I checked `index.ntotal`, added `index.add(doc_embeddings)`, asserted that `index.ntotal == len(corpus_df)`, and updated `retrieve_faiss()` to skip invalid indices. After fixing this, retrieval returned meaningful movie results.

---

## Entry 17

**Task ID:** Task 2 / FAISS vs ChromaDB Comparison  
**Tool:** ChatGPT  
**Prompt:** “No I do wanna compare both Faiss and Chroma, provide me based on this.”  
**Output summary:** ChatGPT provided code to build both FAISS and ChromaDB using the same embeddings, define `retrieve_faiss()` and `retrieve_chroma()`, measure query latency over 20 queries, compute disk size, and create a comparison table.  
**Verified/changed:** I kept both vector databases for comparison but decided to continue with FAISS for the final pipeline. My comparison showed FAISS indexing was much faster, while ChromaDB stored metadata more conveniently. I also noted that FAISS scores and Chroma distances are not directly comparable.

---

## Entry 18

**Task ID:** Task 3 / Cross-Encoder Re-ranking  
**Tool:** ChatGPT  
**Prompt:** “Now I need re-ranking using cross encoders.”  
**Output summary:** ChatGPT provided code to load `cross-encoder/ms-marco-MiniLM-L-6-v2`, implement a `rerank(query, candidates, n=5)` function, and create a FAISS retrieval + re-ranking pipeline. It also explained that the cross-encoder jointly reads the query and candidate document, unlike the bi-encoder.  
**Verified/changed:** I used FAISS as the final retriever and kept ChromaDB only for comparison. I ran re-ranking on example queries such as “Who directed Toy Story?” and observed that re-ranking moved the original `Toy Story` above `Toy Story 3`, which directly demonstrated a sequel-confusion fix.

 ---

## Entry 19

**Task ID:** Task 4 / Local LLM Generation  
**Tool:** ChatGPT  
**Prompt:** “Now lets continue with generation. We'll continue from this code.”  
**Output summary:** ChatGPT provided code to load `Qwen/Qwen2.5-3B-Instruct` with 4-bit quantization, build context from the top-5 re-ranked documents, define a strict grounded system prompt, implement `generate_answer()`, and create an end-to-end RAG generation function using FAISS retrieval, cross-encoder re-ranking, and Qwen generation.  
**Verified/changed:** I used Qwen as the local generator because hosted APIs are prohibited. I generated answers only from re-ranked context, not from raw bi-encoder results, to avoid sequel-confusion errors.

---

## Entry 20

**Task ID:** Task 4 / Query Design  
**Tool:** ChatGPT  
**Prompt:** I asked whether these questions were good: “Which film is the mostly liked series in 21th century?” and “Which film is the most liked film in the 21th century?”  
**Output summary:** ChatGPT explained that these are not good normal RAG questions because they are ambiguous and require corpus-level aggregation over ratings or popularity, which the current `document_text` does not include. It suggested using them only as possible failure diary examples for metadata absence or aggregation failure.  
**Verified/changed:** I avoided using these questions as normal generation test queries. I kept my test queries focused on answerable movie-level questions such as director, plot, and franchise-confusion queries.

---

## Entry 21

**Task ID:** Task 5.3 / RAGAS Debugging  
**Tool:** ChatGPT  
**Prompt:** I pasted RAGAS logs showing repeated `max_new_tokens` warnings, timeout errors, and parsing failures, and asked what the error meant.  
**Output summary:** ChatGPT explained that the `max_new_tokens` warning was harmless, but the real problem was RAGAS timing out and the local Qwen judge failing to produce the exact structured output expected by RAGAS. It suggested reducing judge output length, increasing timeout, using `max_workers=1`, shortening contexts, and using `raise_exceptions=False`.  
**Verified/changed:** I reduced `max_new_tokens`, shortened contexts, used a single worker, and treated parser errors as an issue to document rather than ignore.

---

## Entry 22

**Task ID:** Task 5.3 / RAGAS Runtime and Worker Count  
**Tool:** ChatGPT  
**Prompt:** “Is this normal the process its too slow?” and then “Can’t we increase max worker?”  
**Output summary:** ChatGPT explained that RAGAS evaluation with a local judge model can be slow, but many parser errors are not ideal. It also explained that increasing `max_workers` is risky on a single Colab GPU because parallel judge calls compete for the same GPU memory and may worsen OOM, timeouts, and parsing errors.  
**Verified/changed:** I kept `max_workers=1` for stability, considered testing `max_workers=2` only after a stable run, and documented that local RAGAS judging was much slower and noisier than retrieval/generation.

