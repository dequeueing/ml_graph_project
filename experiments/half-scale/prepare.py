"""Generate the half-scale manifests using the pinned GraphDO checkout."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from functools import cache
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_benchmark import (UPSTREAM, GraphDescriptionGenerator, edge_set, evaluate_answer,
                               generate_task, get_examples, nx, save_jsonl, upstream_functions)
from evaluator import build_graph


def hamilton_path(graph):
    """Find a full path for the grader sanity check, if the graph has one."""
    @cache
    def extend(node, remaining):
        if not remaining:
            return (node,)
        for neighbor in graph[node]:
            if neighbor in remaining:
                path = extend(neighbor, remaining - {neighbor})
                if path:
                    return (node, *path)
        return ()

    nodes = frozenset(graph)
    for start in graph:
        path = extend(start, nodes - {start})
        if path:
            return path
    return ()


def validate_grader(case):
    """Check representative inputs without editing the upstream grader."""
    task, data, query = (case[key] for key in ("task", "graph_data", "query_params"))
    if task == "node_classification":
        node = str(query["test_node"])
        label = data["node_labels"][node]
        answer = f"The label is {label}."
    else:
        graph = build_graph(data, case["is_weighted"], case["is_directed"])
        if task == "connectivity":
            answer = "The answer is yes." if nx.has_path(graph, query["source"], query["target"]) else "The answer is no."
        elif task == "cycle":
            answer = "There is a cycle." if nx.cycle_basis(graph) else "There is no cycle."
        elif task == "shortest_path":
            source, target = query["source"], query["target"]
            path = nx.shortest_path(graph, source, target, weight="weight")
            answer = f"The shortest path from node {source} to node {target}: {path}."
        elif task == "topology":
            answer = f"The solution is {list(nx.topological_sort(graph))}."
        elif task == "hamilton":
            path = hamilton_path(graph)
            if not path:
                # The public Hamilton scorer only accepts valid paths, even if none exists.
                return "hamilton_no_path"
            answer = f"The path is {list(path)}."
        else:
            raise ValueError(task)
    if evaluate_answer(task, answer, data, case["is_weighted"], case["is_directed"], query) != 1:
        raise ValueError(f"Upstream grader failed reference answer for {task}/{case['graph_id']}")
    return "reference_accepted"


def prepare(group):
    plan = json.loads((HERE / "experiment.json").read_text())
    settings = plan[group]
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != plan["upstream_commit"]:
        raise ValueError(f"Wrong upstream commit {commit}")
    output = HERE / "benchmark" / group
    if output.exists():
        raise FileExistsError(output)

    make_prompt, make_questions = upstream_functions()
    random.seed(plan["seed"])
    rng = random.Random(plan["seed"])
    orders = plan["orders"]
    n = settings["graphs_per_group"]
    cell_count = {}
    grader_checks = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{group}-") as tmp:
        stage = Path(tmp) / "manifest"
        graph_root = stage / "graphs"
        stage.mkdir()
        generator = GraphDescriptionGenerator(str(graph_root))
        manifest_path = stage / "manifest.jsonl"
        with manifest_path.open("w") as manifest, (stage / "build.log").open("w") as log:
            for task in settings["tasks"]:
                graphs = generate_task(task, n, rng, settings.get("datasets"))
                if group == "node":
                    found = Counter((g["dataset"], g["sampling"]) for g in graphs)
                    expected = {(d, s): n for d in settings["datasets"] for s in settings["samplings"]}
                    if found != expected or any(g["num_nodes"] != 50 for g in graphs):
                        raise ValueError(f"Unexpected node-classification sample: {found}")
                elif len(graphs) != n:
                    raise ValueError(f"{task}: expected {n} graphs, found {len(graphs)}")
                save_jsonl(graphs, str(graph_root / f"{task}.jsonl"))
                with redirect_stdout(log):
                    descriptions = generator.generate_task_descriptions(task, orders=orders)
                questions = make_questions(descriptions)
                if len(questions) != len(graphs) or len({q["graph_id"] for q in questions}) != len(graphs):
                    raise ValueError(f"{task}: expected one question per graph")
                examples = None if group == "node" else get_examples(task)
                for question in questions:
                    graph = question["graph_data"]
                    cell_group = task if group == "traditional" else f"{graph['dataset']}/{graph['sampling']}"
                    cell_count[cell_group] = cell_count.get(cell_group, 0) + 1
                    grader_checks[validate_grader(question)] += 1
                    ordered = question.pop("descriptions")
                    expected_edges = edge_set([edge[:2] for edge in graph["edges"]], question["is_directed"])
                    for order in orders:
                        variant = ordered[order]
                        if "error" in variant:
                            raise ValueError(f"{task}/{question['graph_id']}/{order}: {variant['error']}")
                        if edge_set(variant["ordered_edges"], question["is_directed"]) != expected_edges:
                            raise ValueError(f"{task}/{question['graph_id']}/{order}: changed edges")
                        for style in settings["prompt_styles"]:
                            prompt = make_prompt(None, variant["description"], question["question"], style, examples)
                            for repeat in range(1, settings["repeats"] + 1):
                                case_id = f"{task}/{question['question_id']}/{order}/{style}/r{repeat}"
                                case = dict(question, case_id=case_id, group=cell_group, order=order,
                                            prompt_style=style, repeat=repeat, prompt=prompt)
                                manifest.write(json.dumps(case, ensure_ascii=False) + "\n")
        count = sum(cell_count.values()) * len(orders) * len(settings["prompt_styles"]) * settings["repeats"]
        expected_total = {"traditional": 17500, "node": 2250}[group]
        if count != expected_total:
            raise ValueError(f"Expected {expected_total} cases, got {count}")
        requirements = ["networkx", "numpy", "scipy"]
        if group == "node":
            requirements += ["torch", "torch-geometric", "littleballoffur"]
        provenance = dict(settings, group=group, orders=orders, models=plan["models"],
                          seed=plan["seed"], upstream_commit=commit, graph_groups=cell_count,
                          grader_checks=grader_checks, prompt_count=count,
                          planned_requests=count * len(plan["models"]),
                          packages={name: version(name) for name in requirements},
                          manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        (stage / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        stage.rename(output)
    print(f"{group}: {count} cases, {provenance['planned_requests']} calls; {grader_checks}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("traditional", "node"))
    prepare(parser.parse_args().group)
