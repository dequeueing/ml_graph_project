"""Recompute RQ2 from the final per-question correctness scores.

Run locally with: python3 analysis/analyze_rq2.py
The input contains audited final scores and their provenance.
No model calls or remote writes are made. Variances are descriptive population
variances of accuracy percentages across five orders within one prompt style.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, pvariance


HERE = Path(__file__).resolve().parent
MODELS = ["deepseek-v4.1-flash", "glm-5.3", "gpt-6-luna"]
TASKS = ["connectivity", "cycle", "shortest_path", "hamilton", "topology"]
PROMPTS = ["zero_shot", "zero_shot_cot", "few_shot", "cot", "cot_bag"]
ORDERS = ["random", "bfs", "dfs", "pagerank", "ppr"]


def compare(a, b):
    if abs(a - b) < 1e-10:
        return "equal"
    return "lower" if b < a else "higher"


def main():
    snapshot = json.loads((HERE / "audited_final_scores.json").read_text())
    previous = json.loads((HERE / "rq1_insights.json").read_text())
    assert snapshot["manifest_sha256"] == previous["manifest_sha256"]
    grouped = defaultdict(lambda: defaultdict(dict))
    for model in MODELS:
        scores = snapshot["models"][model]["scores"]
        assert len(scores) == 15950 and len(dict(scores)) == 15950
        for case_id, correct in scores:
            task, graph_question, order, prompt, repeat = case_id.split("/")
            assert task in TASKS and order in ORDERS and prompt in PROMPTS
            assert correct in (0, 1)
            grouped[model, task, prompt][order][graph_question, repeat] = int(correct)

    cells = {}
    for model in MODELS:
        for task in TASKS:
            for prompt in PROMPTS:
                groups = grouped[model, task, prompt]
                keys = set(groups["random"])
                assert len(keys) == (78 if task == "hamilton" else 140) and set(groups) == set(ORDERS)
                assert all(set(groups[order]) == keys for order in ORDERS)
                counts = {order: sum(groups[order].values()) for order in ORDERS}
                accuracies = {order: 100 * counts[order] / len(keys) for order in ORDERS}
                totals = [sum(groups[order][key] for order in ORDERS) for key in keys]
                cells[model, task, prompt] = {
                    "model": model, "task": task, "prompt": prompt,
                    "questions_per_order": len(keys), "correct_by_order": counts,
                    "accuracy_by_order_pct": accuracies,
                    "mean_accuracy_pct": mean(accuracies.values()),
                    "variance_pp2": pvariance(accuracies.values()),
                    "range_pp": max(accuracies.values()) - min(accuracies.values()),
                    "correct_under_all_orders": sum(v == 5 for v in totals),
                    "wrong_under_all_orders": sum(v == 0 for v in totals),
                    "mixed_correctness_across_orders": sum(0 < v < 5 for v in totals),
                }

    # Independently cross-check the new case-level import against all RQ1 pairs.
    assert len(previous["paired_comparisons"]) == 300
    for row in previous["paired_comparisons"]:
        groups = grouped[row["model"], row["task"], row["prompt"]]
        base, other = groups["random"], groups[row["order"]]
        assert sum(base.values()) == row["baseline_correct"]
        assert sum(other.values()) == row["correct"]
        assert sum(other[k] > base[k] for k in base) == row["gain"]
        assert sum(other[k] < base[k] for k in base) == row["loss"]

    task_summary = []
    for model in MODELS:
        for task in TASKS:
            rows = [cells[model, task, prompt] for prompt in PROMPTS]
            task_summary.append({
                "model": model, "task": task,
                "mean_variance_pp2": mean(r["variance_pp2"] for r in rows),
                "mean_accuracy_pct": mean(r["mean_accuracy_pct"] for r in rows),
                "mean_range_pp": mean(r["range_pp"] for r in rows),
            })

    rankings = []
    for model in MODELS:
        for prompt in PROMPTS:
            values = {task: cells[model, task, prompt]["variance_pp2"] for task in TASKS}
            maximum = max(values.values())
            rankings.append({
                "model": model, "prompt": prompt,
                "tasks_descending": sorted(TASKS, key=values.get, reverse=True),
                "highest_tasks": [t for t in TASKS if abs(values[t] - maximum) < 1e-10],
                "binary_tasks_both_below_nonbinary": max(values[t] for t in TASKS[:2])
                < min(values[t] for t in TASKS[2:]),
            })

    contrasts = []
    for before, after in [("zero_shot", "zero_shot_cot"),
                          ("few_shot", "cot"),
                          ("zero_shot_cot", "cot"), ("cot", "cot_bag")]:
        rows = []
        counts = dict(lower=0, equal=0, higher=0)
        for model in MODELS:
            for task in TASKS:
                a, b = cells[model, task, before], cells[model, task, after]
                direction = compare(a["variance_pp2"], b["variance_pp2"])
                counts[direction] += 1
                rows.append({
                    "model": model, "task": task, "variance_direction": direction,
                    "before_variance_pp2": a["variance_pp2"],
                    "after_variance_pp2": b["variance_pp2"],
                    "before_mean_accuracy_pct": a["mean_accuracy_pct"],
                    "after_mean_accuracy_pct": b["mean_accuracy_pct"],
                })
        contrasts.append({"before": before, "after": after,
                          "variance_changes": counts, "comparisons": rows})

    findings = {
        "binary_tasks_least_variable_settings": sum(r["binary_tasks_both_below_nonbinary"]
                                                     for r in rankings),
        "model_prompt_settings": len(rankings),
        "shortest_path_highest_settings": sum("shortest_path" in r["highest_tasks"]
                                               for r in rankings),
        "zero_shot_highest": {r["model"]: r["highest_tasks"] for r in rankings
                              if r["prompt"] == "zero_shot"},
        "zero_shot_cot_highest": {r["model"]: r["highest_tasks"] for r in rankings
                                  if r["prompt"] == "zero_shot_cot"},
    }
    output = {
        "provenance": {
            **{k: snapshot[k] for k in ["source", "source_sha256", "manifest_sha256", "created_utc", "scorer_sha256"]},
            "score_snapshot": "analysis/audited_final_scores.json",
            "score_snapshot_sha256": hashlib.sha256((HERE / "audited_final_scores.json").read_bytes()).hexdigest(),
        },
        "analysis_scope": "375 order-level accuracies from 47850 main-analysis correctness scores",
        "variance_definition": "Population variance of five accuracy percentages, within a model/task/prompt cell",
        "inference_scope": "Descriptive. Model-response variation is not isolated from ordering effects.",
        "task_summary": task_summary, "cells": list(cells.values()),
        "rankings": rankings, "prompt_contrasts": contrasts, "findings": findings,
    }
    (HERE / "rq2_insights.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(findings, indent=2))
    for row in contrasts:
        print(row["before"], "->", row["after"], row["variance_changes"])
    for model in MODELS:
        values = {r["task"]: round(r["mean_variance_pp2"], 2)
                  for r in task_summary if r["model"] == model}
        print(model, values)
    print("Cross-checked all 300 RQ1 pairs against the final per-question scores.")


if __name__ == "__main__":
    main()
