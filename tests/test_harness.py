"""Tests for llm-eval-harness.

Run from the project root:  pytest
"""

import json
import sys
from pathlib import Path

import pytest

# Make run_eval importable when pytest runs from the project root or tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402


@pytest.fixture()
def config():
    return {
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


def test_score_answer_all_keywords_present():
    answer = "The Summit Blend is a medium roast that costs $18."
    assert run_eval.score_answer(answer, ["Summit Blend", "medium roast", "$18"]) == 1.0


def test_score_answer_no_keywords_present():
    answer = "I don't know based on the provided information."
    assert run_eval.score_answer(answer, ["Summit Blend", "$18"]) == 0.0


def test_score_answer_partial_keywords():
    # 1 of 2 keywords present -> 0.5. Case must not matter.
    answer = "The summit blend is our signature coffee."
    assert run_eval.score_answer(answer, ["Summit Blend", "$18"]) == 0.5


def test_per_category_aggregation(config):
    rows = [
        {"category": "pricing", "score": 1.0, "latency_seconds": 0.1,
         "input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.01},
        {"category": "pricing", "score": 0.0, "latency_seconds": 0.2,
         "input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.01},
        {"category": "brewing", "score": 1.0, "latency_seconds": 0.1,
         "input_tokens": 10, "output_tokens": 20, "estimated_cost": 0.01},
    ]
    report = run_eval.aggregate_results(rows, config)
    assert report["total_questions"] == 3
    assert report["passed"] == 2
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["categories"]["pricing"]["accuracy"] == 0.5
    assert report["categories"]["brewing"]["accuracy"] == 1.0
    assert report["worst_category"] == "pricing"


def test_cost_math(config):
    # 1000 input tokens at $0.15/1k + 1000 output at $0.60/1k = $0.75.
    cost = run_eval.estimate_cost(1000, 1000, config["pricing"])
    assert cost == pytest.approx(0.75)
    assert run_eval.estimate_cost(0, 0, config["pricing"]) == 0.0


def test_estimate_tokens_scales_with_length():
    assert run_eval.estimate_tokens("x") == 1  # never zero
    assert run_eval.estimate_tokens("a" * 400) == 100
    assert run_eval.estimate_tokens("a" * 800) > run_eval.estimate_tokens("a" * 400)


def _write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def test_empty_dataset_handled_gracefully(tmp_path, config):
    dataset = tmp_path / "empty.jsonl"
    kb = tmp_path / "kb.md"
    _write(dataset, "")
    _write(kb, "## About\nBlue Peak was founded in 2018.\n")
    report = run_eval.run_eval(
        str(dataset), str(kb), config, model="local",
        output_dir=str(tmp_path / "results"),
    )
    assert report["total_questions"] == 0
    assert report["accuracy"] == 0.0
    assert report["worst_category"] is None
    assert (tmp_path / "results" / "report.json").exists()
    assert (tmp_path / "results" / "report.md").exists()
    assert (tmp_path / "results" / "traces.jsonl").exists()


def test_full_run_creates_report_files(tmp_path, config):
    project_root = Path(__file__).resolve().parents[1]
    report = run_eval.run_eval(
        str(project_root / "data" / "eval_set.jsonl"),
        str(project_root / "data" / "kb.md"),
        config,
        model="local",
        output_dir=str(tmp_path / "results"),
    )
    assert report["total_questions"] == 20
    assert report["accuracy"] > 0.5  # the local retriever should do well on its own KB
    for name in ("report.json", "report.md", "traces.jsonl"):
        assert (tmp_path / "results" / name).exists()
    saved = json.loads((tmp_path / "results" / "report.json").read_text())
    assert saved["total_questions"] == 20
    assert "categories" in saved and "worst_category" in saved
    traces = (tmp_path / "results" / "traces.jsonl").read_text().strip().splitlines()
    assert len(traces) == 20
    first = json.loads(traces[0])
    assert {"question", "answer", "score", "latency_seconds"} <= set(first)
