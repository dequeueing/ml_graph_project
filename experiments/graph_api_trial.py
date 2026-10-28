#!/usr/bin/env python3
"""Run a small, deterministic cycle-detection trial against the local LLM APIs."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
GRAPH_REPO = ROOT / "talk-like-a-graph"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GRAPH_REPO))

from talk_like_a_graph import graph_generators, graph_text_encoders  # noqa: E402
import query_llms  # noqa: E402


PROFILES = tuple(
    p.strip()
    for p in os.environ.get("GRAPH_TRIAL_PROFILES", "claude-glm,claude-ds,codex").split(",")
    if p.strip()
)
MAX_TOKENS = int(os.environ.get("GRAPH_TRIAL_MAX_TOKENS", "512"))
CASES = (
    ("path", 1001),
    ("star", 1002),
    ("complete", 1003),
    ("er", 1004),
    ("er", 1005),
    ("ba", 1006),
    ("ba", 1007),
    ("sbm", 1008),
    ("sfn", 1009),
    ("path", 1010),
)


def has_cycle(graph: nx.Graph) -> bool:
    try:
        nx.find_cycle(graph)
    except nx.NetworkXNoCycle:
        return False
    return True


def make_prompt(graph_text: str) -> str:
    return (
        "You are solving a basic graph reasoning question. "
        "The graph is undirected. Read the complete graph description. "
        "Determine whether it contains at least one cycle. "
        "Answer with exactly one word: Yes or No. Do not add an explanation.\n\n"
        "Graph description:\n"
        + graph_text
        + "\nQuestion: Is there a cycle in this graph?\nAnswer:"
    )


def normalize_answer(text: str):
    match = re.search(r"\b(yes|no)\b", text.lower())
    return match.group(1) == "yes" if match else None


def ask_one(profile: str, prompt: str):
    started = time.perf_counter()
    try:
        result = query_llms.query(profile, prompt, max_tokens=MAX_TOKENS, timeout=120)
        text = result["text"].strip()
        return {
            "status": "ok",
            "answer_text": text,
            "predicted_cycle": normalize_answer(text),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # Keep one failed provider from stopping the trial.
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def main():
    records = []
    for index, (algorithm, seed) in enumerate(CASES, start=1):
        graph = graph_generators.generate_graphs(
            number_of_graphs=1,
            algorithm=algorithm,
            directed=False,
            random_seed=seed,
        )[0]
        graph_text = graph_text_encoders.encode_graph(graph, graph_encoder="incident")
        record = {
            "graph_id": index,
            "algorithm": algorithm,
            "seed": seed,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "ground_truth_cycle": has_cycle(graph),
            "encoding": "incident",
            "prompt": make_prompt(graph_text),
            "providers": {},
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROFILES)) as pool:
            futures = {
                pool.submit(ask_one, profile, record["prompt"]): profile
                for profile in PROFILES
            }
            for future in concurrent.futures.as_completed(futures):
                record["providers"][futures[future]] = future.result()
        records.append(record)
        summary = ", ".join(
            f"{p}={record['providers'][p].get('answer_text', record['providers'][p].get('status'))}"
            for p in PROFILES
        )
        print(
            f"graph {index:02d} {algorithm:8} nodes={record['nodes']:2d} "
            f"edges={record['edges']:3d} truth={record['ground_truth_cycle']} | {summary}",
            flush=True,
        )

    output = ROOT / "experiments" / "graph_api_trial_results.json"
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {output}")

    for profile in PROFILES:
        values = [r["providers"].get(profile, {}) for r in records]
        valid = [v for v in values if v.get("status") == "ok"]
        correct = [v for r, v in zip(records, values) if v.get("predicted_cycle") == r["ground_truth_cycle"]]
        print(f"{profile}: {len(correct)}/{len(valid)} correct, {len(valid)}/{len(records)} responses")


if __name__ == "__main__":
    main()
