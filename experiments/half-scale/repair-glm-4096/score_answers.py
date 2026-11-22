"""Recheck explicit saved answers using the original graph and question."""

from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
import sys

import networkx as nx

REPAIR = Path(__file__).resolve().parent
ROOT = REPAIR.parent
sys.path.insert(0, str(ROOT.parent))
from run_experiment import load_benchmark

LABEL = re.compile(r"^\s*(yes|no)\b", re.I)
FINAL_LABEL = re.compile(r"\b(?:(?:final|the) )?answer(?:\s+is)?\s*[:=]?\s*(yes|no)\b", re.I)
ARROWS = re.compile(r"\d+(?:\s*->\s*\d+)+")
PATH_CUE = re.compile(r"\b(?:path|order|ordering|sorting|answer)(?:\s+is)?\s*:\s*", re.I)


def normalized(text):
    for old, new in (("**", ""), ("__", ""), ("`", ""), (r"\(", ""),
                     (r"\)", ""), (r"\[", ""), (r"\]", ""),
                     (r"\rightarrow", "->"), (r"\to", "->"), ("→", "->")):
        text = text.replace(old, new)
    return text.strip()


def explicit_label(text):
    final = list(FINAL_LABEL.finditer(text))
    match = final[-1] if final else (LABEL.match(text.splitlines()[-1]) or LABEL.match(text))
    return None if match is None else match[1].lower() == "yes"


def numeric_list(line):
    parts = re.split(r"\s*(?:->|,)\s*|\s+", line.strip(" []."))
    if all(part.isascii() and part.isdecimal() for part in parts):
        return tuple(map(int, parts))
    return None


def path_answer(line):
    standalone = numeric_list(line)
    if standalone is not None:
        return standalone
    cue = PATH_CUE.search(line)
    if cue:
        labeled = numeric_list(line[cue.end():])
        if labeled is not None:
            return labeled
    arrows = ARROWS.findall(line)
    if len(arrows) == 1 and re.search(r"path|order|weight|length|answer|correction", line, re.I):
        return tuple(map(int, re.findall(r"\d+", arrows[0])))
    return None


@lru_cache(maxsize=None)
def graph_for(task, n, edges):
    graph = nx.DiGraph() if task == "topology" else nx.Graph()
    graph.add_nodes_from(range(n))
    if task == "shortest_path":
        graph.add_weighted_edges_from(edges)
    else:
        graph.add_edges_from(edges)
    return graph


@lru_cache(maxsize=None)
def hamiltonian(graph):
    if not nx.is_connected(graph):
        return False

    @lru_cache(maxsize=None)
    def walk(node, visited):
        if visited.bit_count() == len(graph):
            return True
        return any(walk(v, visited | (1 << v)) for v in graph[node] if not (visited & (1 << v)))

    return any(walk(v, 1 << v) for v in graph)


def adjudicate(case, answer):
    text = normalized(answer)
    if not text:
        return None
    task = case["task"]
    data = case["graph_data"]
    graph = graph_for(task, data["num_nodes"], tuple(map(tuple, data["edges"])))
    last = text.splitlines()[-1]
    label = explicit_label(text)
    if task == "cycle":
        return None if label is None else label == bool(nx.cycle_basis(graph))
    if task == "connectivity":
        query = case["query_params"]
        return None if label is None else label == nx.has_path(graph, query["source"], query["target"])
    if task == "hamilton" and (label is False or re.search(r"\bthere is no hamiltonian path\b", last, re.I)):
        return not hamiltonian(graph)

    path = path_answer(last)
    if path is None:
        return None
    vertices = set(path)
    if vertices - set(graph) or len(vertices) != len(path):
        return False
    if task == "topology":
        pos = {v: i for i, v in enumerate(path)}
        if set(graph) - vertices > set(nx.isolates(graph)):
            return False
        if any(u not in pos or v not in pos or pos[u] >= pos[v] for u, v in graph.edges):
            return False
        return True
    if any(not graph.has_edge(u, v) for u, v in zip(path, path[1:])):
        return False
    if task == "hamilton":
        return vertices == set(graph)
    query = case["query_params"]
    source, target = query["source"], query["target"]
    if (path[0], path[-1]) != (source, target):
        return False
    return nx.path_weight(graph, path, "weight") == nx.shortest_path_length(
        graph, source, target, weight="weight")


def read_records(path, manifest):
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT manifest_hash FROM metadata").fetchone() == (manifest,)
        return {key: json.loads(value) for key, value in conn.execute(
            "SELECT case_id, payload FROM results WHERE status='ok'")}


def main():
    assert path_answer("5 -> 6 -> 2 -> 0, with total weight 19.") == (5, 6, 2, 0)
    assert path_answer("One valid topological sorting is: 2, 3, 6, 5.") == (2, 3, 6, 5)
    assert path_answer("The total weight is 19.") is None
    assert explicit_label("Yes. Explanation. The answer is No.") is False
    config, cases = load_benchmark(ROOT / "benchmark" / "traditional")
    manifest = config["manifest_sha256"]
    expected = {case["case_id"] for case in cases}
    assert len(expected) == len(cases) == 17500
    result = {"manifest_sha256": manifest, "models": {}}
    for model in ("deepseek-v4.1-flash", "gpt-6-luna", "glm-5.3"):
        base = read_records(ROOT / "runs/traditional" / f"{model}.sqlite3", manifest)
        assert base.keys() == expected
        patched = (read_records(REPAIR / "runs/traditional" / f"{model}.sqlite3", manifest)
                   if model == "glm-5.3" else {})
        assert all(base[key].get("finish_reason") == "max_tokens" and row["text"].strip()
                   and row.get("finish_reason") != "max_tokens" for key, row in patched.items())
        cells = defaultdict(Counter)
        paired = []
        for case in cases:
            key = case["case_id"]
            row = patched.get(key, base[key])
            recorded = bool(row["correct"])
            checked = adjudicate(case, row["text"])
            score = recorded if checked is None else checked
            counts = cells[case["task"], case["order"], case["prompt_style"]]
            counts.update(n=1, old_release=bool(base[key]["correct"]), release=recorded,
                          score=score, verified=checked is not None, gained=score and not recorded,
                          lost=recorded and not score, blank=not row["text"].strip(), replaced=key in patched)
            paired.append((key, int(score)))
        assert len(cells) == 125 and all(counts["n"] == 140 for counts in cells.values())
        result["models"][model] = {
            "repaired": len(patched),
            "cells": [dict(task=task, order=order, prompt=prompt, **counts)
                      for (task, order, prompt), counts in sorted(cells.items())],
            "scores": paired,
        }
        print(model, f"replaced {len(patched)} of 5991", flush=True)
        for task in config["tasks"]:
            total = Counter()
            for (cell_task, _, _), counts in cells.items():
                if task == cell_task:
                    total.update(counts)
            print(" ", task, dict(total), flush=True)
        for order in config["orders"]:
            total = Counter()
            for (_, cell_order, _), counts in cells.items():
                if order == cell_order:
                    total.update(counts)
            print(" ", order, [round(100 * total[k] / total["n"], 2)
                                for k in ("old_release", "release", "score")], flush=True)
    out = REPAIR / "scoring-audit.json"
    temp = out.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result))
    temp.replace(out)


if __name__ == "__main__":
    main()
