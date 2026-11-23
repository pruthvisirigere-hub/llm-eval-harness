"""llm-eval-harness: a small LLM evaluation + observability harness.

What it does
------------
1. Loads a knowledge base (data/kb.md) and an eval set (data/eval_set.jsonl).
2. Answers every question with a "model":
   - ``local`` (default): TF-IDF retrieval over the KB, no API key needed.
   - ``openrouter``: calls an OpenRouter chat model with the KB as context.
     Needs the OPENROUTER_API_KEY environment variable.
3. Scores each answer by keyword match against the expected keywords.
4. Records latency, estimated tokens and estimated cost per question.
5. Writes results/report.json, results/report.md and results/traces.jsonl.

Run it:
    python run_eval.py
    python run_eval.py --model openrouter
    python run_eval.py --dataset data/eval_set.jsonl --output-dir results
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path):
    """Load config.yaml. Missing keys fall back to the defaults below."""
    defaults = {
        "retrieval": {"top_k": 1},
        "scoring": {"pass_threshold": 0.5},
        "pricing": {
            "input_price_per_1k": 0.15,
            "output_price_per_1k": 0.60,
            "currency": "USD",
        },
        "openrouter": {
            "model": "openai/gpt-4o-mini",
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "max_context_chars": 6000,
        },
        "run": {"default_model": "local"},
    }
    with open(path, encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    # Shallow-merge one level deep so a partial config still works.
    for section, values in defaults.items():
        if section in user_config and isinstance(user_config[section], dict):
            merged = dict(values)
            merged.update(user_config[section])
            defaults[section] = merged
    return defaults


def load_kb(path):
    """Split a Markdown KB into chunks, one per ## / ### heading section.

    Returns a list of {"title": str, "text": str}. Text before the first
    heading is dropped, because it has no section to belong to.
    """
    text = Path(path).read_text(encoding="utf-8")
    chunks = []
    current_title = None
    current_lines = []
    for line in text.splitlines():
        match = re.match(r"^(#{2,3})\s+(.*)", line)
        if match:
            if current_title is not None:
                chunks.append(
                    {"title": current_title, "text": "\n".join(current_lines).strip()}
                )
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        chunks.append(
            {"title": current_title, "text": "\n".join(current_lines).strip()}
        )
    return [c for c in chunks if c["text"]]


def load_dataset(path):
    """Load a JSONL eval set. Each line needs question, expected_keywords,
    category. Blank lines are skipped; a bad line raises a clear error."""
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: not valid JSON: {exc}") from exc
            for field in ("question", "expected_keywords", "category"):
                if field not in item:
                    raise ValueError(
                        f"{path}:{line_number}: missing required field '{field}'"
                    )
            questions.append(item)
    return questions


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_retriever(chunks):
    """Fit a TF-IDF vectorizer over the KB chunks. Imported lazily so that
    --help and the OpenRouter path do not pay the sklearn import cost."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return vectorizer, matrix


def answer_local(question, chunks, vectorizer, matrix, top_k=1):
    """Extractive answer: return the most relevant KB chunk(s) for the
    question, highest TF-IDF cosine similarity first."""
    from sklearn.metrics.pairwise import cosine_similarity

    if not chunks:
        return "No knowledge base loaded."
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix)[0]
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    best = ranked[: max(top_k, 1)]
    if scores[best[0]] == 0:
        return "I don't know based on the provided information."
    return "\n\n".join(chunks[i]["text"] for i in best)


def answer_openrouter(question, kb_text, config):
    """Answer via an OpenRouter chat model with the KB pasted as context."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it first, e.g.\n"
            "  export OPENROUTER_API_KEY=sk-or-..."
        )
    settings = config["openrouter"]
    context = kb_text[: settings["max_context_chars"]]
    payload = {
        "model": settings["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the information below. "
                    "If the information does not contain the answer, say so.\n\n"
                    + context
                ),
            },
            {"role": "user", "content": question},
        ],
    }
    request = urllib.request.Request(
        settings["base_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Scoring, tokens, cost
# ---------------------------------------------------------------------------

def score_answer(answer, expected_keywords):
    """Fraction of expected keywords found in the answer (case-insensitive
    substring match). 1.0 = every keyword present, 0.0 = none present."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def estimate_tokens(text):
    """Rough token estimate: ~4 characters per token. Good enough for cost
    tracking in a harness; do not use it for billing decisions."""
    return max(1, len(text) // 4)


def estimate_cost(input_tokens, output_tokens, pricing):
    """Cost in the configured currency, from per-1k-token prices."""
    return (
        input_tokens * pricing["input_price_per_1k"] / 1000
        + output_tokens * pricing["output_price_per_1k"] / 1000
    )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def aggregate_results(per_question, config):
    """Roll per-question rows up into overall + per-category metrics."""
    threshold = config["scoring"]["pass_threshold"]
    total = len(per_question)
    passed = sum(1 for row in per_question if row["score"] >= threshold)
    mean_score = (
        sum(row["score"] for row in per_question) / total if total else 0.0
    )
    total_latency = sum(row["latency_seconds"] for row in per_question)
    total_input_tokens = sum(row["input_tokens"] for row in per_question)
    total_output_tokens = sum(row["output_tokens"] for row in per_question)
    total_cost = sum(row["estimated_cost"] for row in per_question)

    by_category = {}
    for row in per_question:
        cat = by_category.setdefault(
            row["category"], {"total": 0, "passed": 0, "score_sum": 0.0}
        )
        cat["total"] += 1
        cat["passed"] += 1 if row["score"] >= threshold else 0
        cat["score_sum"] += row["score"]
    categories = {
        name: {
            "total": stats["total"],
            "passed": stats["passed"],
            "accuracy": stats["passed"] / stats["total"] if stats["total"] else 0.0,
            "mean_score": stats["score_sum"] / stats["total"]
            if stats["total"]
            else 0.0,
        }
        for name, stats in sorted(by_category.items())
    }
    worst_category = None
    if categories:
        worst_category = min(
            categories.items(), key=lambda item: (item[1]["accuracy"], item[1]["mean_score"])
        )[0]

    return {
        "total_questions": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "mean_score": mean_score,
        "avg_latency_seconds": total_latency / total if total else 0.0,
        "total_latency_seconds": total_latency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_estimated_cost": total_cost,
        "currency": config["pricing"]["currency"],
        "pass_threshold": threshold,
        "categories": categories,
        "worst_category": worst_category,
    }


def render_report_md(report, per_question):
    """Human-readable Markdown report: headline metrics, per-category table,
    per-question table, and a findings section naming the worst category."""
    lines = ["# Eval Report", ""]
    lines.append(f"- **Questions:** {report['total_questions']}")
    lines.append(f"- **Passed:** {report['passed']} / {report['total_questions']}")
    lines.append(f"- **Accuracy:** {report['accuracy']:.1%}")
    lines.append(f"- **Mean keyword score:** {report['mean_score']:.2f}")
    lines.append(
        f"- **Avg latency:** {report['avg_latency_seconds']:.2f}s per question"
    )
    lines.append(
        f"- **Total tokens:** {report['total_input_tokens']} in / "
        f"{report['total_output_tokens']} out"
    )
    lines.append(
        f"- **Total estimated cost:** "
        f"{report['total_estimated_cost']:.4f} {report['currency']}"
    )
    lines.append(f"- **Pass threshold:** score >= {report['pass_threshold']}")
    lines.append("")
    lines.append("## Per-category accuracy")
    lines.append("")
    lines.append("| Category | Questions | Passed | Accuracy | Mean score |")
    lines.append("|----------|----------:|-------:|---------:|-----------:|")
    for name, stats in report["categories"].items():
        lines.append(
            f"| {name} | {stats['total']} | {stats['passed']} | "
            f"{stats['accuracy']:.1%} | {stats['mean_score']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-question results")
    lines.append("")
    lines.append("| # | Category | Score | Latency (s) | Question |")
    lines.append("|---:|----------|------:|------------:|----------|")
    for i, row in enumerate(per_question, start=1):
        question = row["question"].replace("|", "\\|")
        lines.append(
            f"| {i} | {row['category']} | {row['score']:.2f} | "
            f"{row['latency_seconds']:.2f} | {question} |"
        )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if report["total_questions"] == 0:
        lines.append("No questions were evaluated (empty dataset).")
    else:
        worst = report["worst_category"]
        stats = report["categories"][worst]
        lines.append(
            f"- **Weakest category: {worst}** — {stats['passed']}/{stats['total']} "
            f"passed ({stats['accuracy']:.0%}, mean score {stats['mean_score']:.2f}). "
            "Look at the failed questions in this category first: either the "
            "knowledge base is missing the fact, the retriever is picking the "
            "wrong chunk, or the expected keywords are worded differently "
            "from the KB."
        )
        failed = [r for r in per_question if r["score"] < report["pass_threshold"]]
        if failed:
            lines.append(f"- **{len(failed)} question(s) failed.** The worst ones:")
            for row in sorted(failed, key=lambda r: r["score"])[:5]:
                missing = [
                    kw
                    for kw in row["expected_keywords"]
                    if kw.lower() not in row["answer"].lower()
                ]
                lines.append(
                    f"  - \"{row['question']}\" (score {row['score']:.2f}; "
                    f"missing keywords: {', '.join(missing) or 'none'})"
                )
        else:
            lines.append("- **No failures.** Every question passed the threshold.")
        slowest = max(per_question, key=lambda r: r["latency_seconds"])
        lines.append(
            f"- **Slowest question:** \"{slowest['question']}\" at "
            f"{slowest['latency_seconds']:.2f}s."
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------

def run_eval(dataset_path, kb_path, config, model="local", output_dir="results"):
    """Run the full eval. Returns the report dict and writes
    report.json, report.md and traces.jsonl into output_dir."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    questions = load_dataset(dataset_path)
    chunks = load_kb(kb_path)
    kb_text = Path(kb_path).read_text(encoding="utf-8")

    vectorizer = matrix = None
    if model == "local":
        vectorizer, matrix = build_retriever(chunks)

    per_question = []
    traces_path = output / "traces.jsonl"
    with open(traces_path, "w", encoding="utf-8") as trace_file:
        for item in questions:
            question = item["question"]
            started = time.perf_counter()
            if model == "local":
                answer = answer_local(
                    question,
                    chunks,
                    vectorizer,
                    matrix,
                    top_k=config["retrieval"]["top_k"],
                )
                input_tokens = estimate_tokens(question)
            elif model == "openrouter":
                answer = answer_openrouter(question, kb_text, config)
                input_tokens = estimate_tokens(question) + estimate_tokens(
                    kb_text[: config["openrouter"]["max_context_chars"]]
                )
            else:
                raise ValueError(f"Unknown model '{model}'. Use local or openrouter.")
            latency = time.perf_counter() - started

            score = score_answer(answer, item["expected_keywords"])
            output_tokens = estimate_tokens(answer)
            cost = estimate_cost(
                input_tokens, output_tokens, config["pricing"]
            )
            row = {
                "question": question,
                "category": item["category"],
                "expected_keywords": item["expected_keywords"],
                "answer": answer,
                "score": round(score, 4),
                "passed": score >= config["scoring"]["pass_threshold"],
                "latency_seconds": round(latency, 4),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": round(cost, 6),
                "model": model,
            }
            per_question.append(row)
            trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = aggregate_results(per_question, config)
    report["model"] = model
    report["dataset"] = str(dataset_path)
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(
        render_report_md(report, per_question), encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an LLM eval: answer questions, score them, report."
    )
    parser.add_argument("--dataset", default="data/eval_set.jsonl")
    parser.add_argument("--kb", default="data/kb.md")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model",
        choices=["local", "openrouter"],
        default=None,
        help="local = TF-IDF retrieval (no API key); openrouter = API model.",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model = args.model or config["run"]["default_model"]
    try:
        report = run_eval(args.dataset, args.kb, config, model, args.output_dir)
    except FileNotFoundError as exc:
        print(f"Error: file not found — {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Model:      {model}")
    print(f"Questions:  {report['total_questions']}")
    print(f"Passed:     {report['passed']} / {report['total_questions']}")
    print(f"Accuracy:   {report['accuracy']:.1%}")
    print(f"Mean score: {report['mean_score']:.2f}")
    print(f"Avg latency: {report['avg_latency_seconds']:.2f}s")
    print(
        f"Est. cost:  {report['total_estimated_cost']:.4f} "
        f"{report['currency']}"
    )
    if report["worst_category"]:
        print(f"Weakest:    {report['worst_category']}")
    print(f"Reports:    {args.output_dir}/report.json, {args.output_dir}/report.md")
    print(f"Traces:     {args.output_dir}/traces.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
