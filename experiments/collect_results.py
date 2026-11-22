"""Verify the three completed experiment shards and collect paper-aligned statistics."""

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
import json
import re
import sqlite3
import sys

import networkx as nx
import numpy as np

from run_experiment import load_benchmark, prepare_database, summarize, evaluate_answer
from argparse import Namespace

LANES = {
    "deepseek": "deepseek-v4.1-flash",
    "glm": "glm-5.3",
    "luna": "gpt-6-luna",
}
BENCHMARK = Path("benchmark/subset")
RUNS = Path("runs")
CONFIG, CASES = load_benchmark(BENCHMARK)
BY_ID = {case["case_id"]: case for case in CASES}


def completed_rows():
    merged = []
    for lane, model in LANES.items():
        path = (RUNS / f"formal-{lane}.sqlite3").resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            claimed_hash, = db.execute("SELECT manifest_hash FROM metadata").fetchone()
            assert claimed_hash == CONFIG["manifest_sha256"], lane
            entries = db.execute("SELECT case_id, model, status, payload FROM results").fetchall()
        assert {row[0] for row in entries} == BY_ID.keys() and len(entries) == len(CASES), lane
        for case_id, name, status, payload in entries:
            row = json.loads(payload)
            assert row["case_id"] == case_id and row["model"] == name == model
            assert row["status"] == status == "ok", (lane, case_id, row.get("error"))
            case = BY_ID[case_id]
            score = evaluate_answer(case["task"], row["text"], case["graph_data"],
                                    case["is_weighted"], case["is_directed"], case["query_params"])
            assert score == row["correct"], (lane, case_id)
            merged.append((case_id, name, status, payload))
    return merged


def cycle_truth(case):
    graph = nx.Graph()
    graph.add_edges_from(case["graph_data"]["edges"])
    return bool(nx.cycle_basis(graph))


def score_audit(entries):
    grouped = defaultdict(Counter)
    details = []
    for case_id, model, _, payload in entries:
        case = BY_ID[case_id]
        row = json.loads(payload)
        counts = grouped[model, case["task"], case["order"]]
        counts["n"] += 1
        counts["correct"] += row["correct"]
        counts["blank"] += not row["text"].strip()
        counts["truncated"] += row.get("finish_reason") == "max_tokens"
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            counts[key] += row["usage"].get(key, 0) or 0
        if case["task"] == "cycle":
            ground_truth = cycle_truth(case)
            counts["cyclic" if ground_truth else "acyclic"] += 1
            counts["cyclic_correct" if ground_truth else "acyclic_correct"] += row["correct"]
            prefix = re.match(r"^\s*[\*_]{0,2}(yes|no)\b", row["text"], re.I)
            if prefix:
                agrees = (prefix[1].lower() == "yes") == ground_truth
                counts["yes_no_prefix"] += 1
                counts["prefix_agrees_official_wrong"] += agrees and row["correct"] == 0
                if agrees and row["correct"] == 0 and len(details) < 5:
                    details.append(dict(model=model, case_id=case_id, cyclic=ground_truth,
                                        response=row["text"][:240]))
    cells = [dict(model=m, task=t, order=o, **counts) for (m,t,o), counts in grouped.items()]
    return cells, details


def paired_intervals(entries):
    scores = defaultdict(dict)
    for case_id, model, _, payload in entries:
        case = BY_ID[case_id]
        scores[model, case["task"], case["prompt_style"], case["order"]][case["graph_id"]] = json.loads(payload)["correct"]
    draws = np.random.default_rng(5222).integers(0, CONFIG["graphs_per_task"],
                                               size=(10000,CONFIG["graphs_per_task"]))
    comparisons = []
    for model in CONFIG["models"]:
        for task in CONFIG["tasks"]:
            for style in CONFIG["prompt_styles"]:
                for order, baseline in (("bfs","random"),("dfs","random"),("bfs","dfs")):
                    test = scores[model, task, style, order]
                    reference = scores[model, task, style, baseline]
                    graphs = sorted(reference)
                    assert len(graphs) == CONFIG["graphs_per_task"] and set(graphs) == test.keys()
                    changes = np.asarray([test[g] - reference[g] for g in graphs])
                    low, high = np.quantile(changes[draws].mean(axis=1) * 100, [0.025, 0.975])
                    comparisons.append(dict(
                        model=model, task=task, order=order, baseline=baseline,
                        delta_pp=float(changes.mean()*100), ci_low_pp=float(low), ci_high_pp=float(high),
                        improved=int((changes == 1).sum()), worsened=int((changes == -1).sum()),
                    ))
    return comparisons


def main():
    entries = completed_rows()
    merged = RUNS / "formal.sqlite3"
    assert not merged.exists(), "Combined database already exists"
    staging_db = RUNS / ".formal-merging.sqlite3"
    assert not staging_db.exists(), "An earlier merge is present"
    with closing(sqlite3.connect(staging_db)) as db:
        prepare_database(db, CONFIG["manifest_sha256"])
        with db:
            db.executemany("INSERT INTO results VALUES (?,?,?,?)", entries)
    staging_db.rename(merged)
    summarize(Namespace(db=merged, output=RUNS / "formal-summary.json"), CONFIG, CASES)
    cells, examples = score_audit(entries)
    diagnostics = {"cells": cells, "cycle_format_examples": examples, "paired": paired_intervals(entries)}
    (RUNS / "formal-diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"Verified and merged {len(entries)} responses")


if __name__ == "__main__":
    main()
