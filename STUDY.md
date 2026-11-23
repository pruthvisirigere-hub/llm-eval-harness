# STUDY.md — Understanding the LLM Eval Harness

Read this before the code. The whole point of this project is the ideas below;
the code is just those ideas written in Python.

## The picture first

You built a question-answering system (CodeRAG). Someone asks a question, the
system answers. Now the obvious next question: **how do you know whether the
answers are any good?**

You cannot answer that by reading five answers and saying "looks fine". That is
vibes, not engineering. An **eval harness** is the tool that replaces vibes
with numbers:

1. You write a fixed set of questions where you already know the right answer
   (the **eval set** — here, `data/eval_set.jsonl`).
2. You run your system on every question and record what it said.
3. You score each answer automatically against what you expected.
4. You aggregate: overall accuracy, accuracy **per category**, latency, tokens,
   cost.
5. You read the report, find the weakest category, and fix that first.

That loop — measure, find the weak spot, fix, measure again — is the entire
job. Everything else in this repo is plumbing for that loop.

## What "evals" actually are

An eval is a test for an AI system, the same way unit tests are tests for
normal code. The difference: normal code is deterministic (same input, same
output), so a test can say "output must equal exactly this". LLM output is
not deterministic, so evals usually check something looser, like "does the
answer contain the key facts?".

This harness scores with **keyword match**: each question lists the facts the
answer must contain (for example `["Summit Blend", "$18"]`), and the score is
the fraction of those keywords that actually appear in the answer. It is
simple, explainable, and honest about what it measures. It is not perfect —
see "Why accuracy alone lies" below.

Two terms you will hear in interviews:

- **Golden dataset / eval set**: the questions + expected answers. Ours has 20
  questions about a fictional coffee company, in 5 categories.
- **Regression**: re-running the same eval set after every change, to prove you
  did not break what used to work. The harness writes
  `results/report.json` every run, so you can compare runs over time.

## Why accuracy alone lies

One number — "92% accurate" — hides almost everything that matters:

1. **It hides where you fail.** 92% overall can mean 100% on easy questions
   and 40% on pricing questions. If pricing is what your users ask about most,
   your system is bad where it counts. That is why this harness reports
   **per-category accuracy** and names the **worst category** in the report.
   Always fix the worst category first.
2. **It hides how you measured.** A 100% score on 3 toy questions (like the
   old CodeRAG README claimed) means nothing. Accuracy is only as credible as
   the eval set behind it. A good eval set is bigger, covers every category,
   and includes hard questions — not just the ones you know you pass.
3. **It hides the cost of being right.** A model can be 95% accurate and
   unusable because it takes 20 seconds per question or costs $2 per answer.
   That is why the harness records **latency, tokens, and cost** next to every
   score. A real eval answer is always three numbers: quality, speed, cost.
4. **Keyword match can be gamed.** An answer can contain all the keywords and
   still be wrong ("Summit Blend costs $18" vs "Summit Blend does NOT cost
   $18" — both contain the keywords). Keyword scoring is a starting point, not
   the truth. Better scorers exist: an LLM judging the answer, or human
   review of a sample. Know the limit and say it out loud — interviewers
   respect that more than pretending the metric is perfect.

## What latency / cost tracing means

Every question in `results/traces.jsonl` has one line with the question, the
answer, the score, the latency in seconds, the token counts, and the cost.
That file is the **observability** part of this project.

- **Latency** is wall-clock time per question. For the local TF-IDF model it
  is milliseconds; for an API model it is seconds. In production you care
  about the average *and* the slowest question (the report calls it out),
  because users remember the slow one.
- **Tokens** are the rough unit API providers bill by. The harness estimates
  tokens as `characters / 4` — a standard rough rule. It is an estimate, not a
  bill; the real count comes from the provider.
- **Cost** is `tokens × price per 1k tokens` from `config.yaml`. For the local
  model the cost is hypothetical — the local model is free, so the report
  shows what the same traffic *would* cost at API prices. That is the number
  you use when deciding "is it worth calling the expensive model for this
  question, or is the cheap one good enough?". That trade-off — quality vs
  cost vs speed — is the real engineering decision behind every LLM product.

## What a good vs bad eval set looks like

Bad eval set:

- 3 questions, all easy, all from one category.
- Expected answers written after seeing the system's output (so they match by
  construction).
- No hard questions, no questions the system should refuse or say "I don't
  know" to.

Good eval set:

- Enough questions per category to make a percentage meaningful (at least 4–5
  per category; ours has 4 per category across 5 categories).
- Questions written from the knowledge base, not from the system's answers.
- A mix of easy lookups ("how much is X?") and questions that need the
  retriever to pick the right chunk.
- Edge cases: a question whose answer is not in the KB at all (the system
  should say it does not know, not invent an answer).

If you extend this project, the single highest-value change is growing the
eval set from 20 to 50+ questions and adding a few unanswerable ones.

## How the pieces map to the code

| Idea | Where it lives |
|---|---|
| Eval set (questions + expected facts) | `data/eval_set.jsonl` |
| Knowledge base the model answers from | `data/kb.md` |
| The "model": TF-IDF retrieval, extractive answers | `answer_local()` in `run_eval.py` |
| The "model": OpenRouter API comparison | `answer_openrouter()` in `run_eval.py` |
| Scoring (keyword match) | `score_answer()` |
| Token + cost estimates | `estimate_tokens()`, `estimate_cost()` |
| Per-category aggregation, worst category | `aggregate_results()` |
| The readable report | `render_report_md()` → `results/report.md` |
| The trace log (observability) | `results/traces.jsonl` |
| Prices, thresholds | `config.yaml` |

## 5 interview questions you should be able to answer about this

1. **"Why did you build an eval harness instead of just testing the model by hand?"**
   Hand-testing five answers is vibes. A harness gives repeatable numbers, so
   every change can be proven better or worse, and regressions get caught.
2. **"Your accuracy is X%. Why should I trust that number?"**
   Talk about the eval set: 20 questions, 5 categories, written from the KB
   before running anything. Then be honest: keyword match is a proxy, it can
   be gamed, and the set is still small — the next step is a bigger set and an
   LLM judge.
3. **"What was the weakest category, and what did you do about it?"**
   Read it from the report's Findings section. The answer is never "nothing,
   it was fine" — there is always a weakest category, and the fix is either
   better retrieval, a better KB, or better expected keywords.
4. **"Why track latency and cost next to accuracy?"**
   Because accuracy alone does not ship a product. A slower or more expensive
   model has to earn its keep with clearly better quality, and you can only
   judge that trade-off if you measure all three together.
5. **"How would you extend this to evaluate a real RAG system like CodeRAG?"**
   Point the harness at CodeRAG's `/api/ask/` endpoint as a third model,
   grow the eval set to questions over a real codebase, and add retrieval
   metrics (was the right file retrieved?) alongside answer metrics. The
   harness already has the shape for that — only the model function changes.
