"""Freeze GraphDO cases and prompts for the models under test."""

import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "GraphDO"
sys.path.insert(0, str(UPSTREAM))

from description_generator import GraphDescriptionGenerator
from examples import get_examples
from graph_generator import generate_task, save_jsonl
from evaluator import evaluate_answer
import networkx as nx


def upstream_functions():
    """Use the exact prompt and question functions without importing local GPU inference."""
    prompt_source = UPSTREAM / "graphdo.py"
    prompt_tree = ast.parse(prompt_source.read_text())
    graphdo_class = next(node for node in prompt_tree.body if isinstance(node, ast.ClassDef)
                         and node.name == "GraphDO")
    prompt_function = next(node for node in graphdo_class.body
                           if isinstance(node, ast.FunctionDef) and node.name == "_construct_prompt")
    question_source = UPSTREAM / "main.py"
    question_tree = ast.parse(question_source.read_text())
    question_function = next(node for node in question_tree.body
                             if isinstance(node, ast.FunctionDef)
                             and node.name == "extract_questions_from_descriptions")
    namespace = {"Any": Any, "Dict": Dict, "List": List}
    for source, function in ((prompt_source, prompt_function), (question_source, question_function)):
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(source), "exec"), namespace)
    return namespace["_construct_prompt"], namespace["extract_questions_from_descriptions"]


def validate_answer_path(question):
    """Verify that the unchanged upstream grader accepts a known correct answer."""
    data = question["graph_data"]
    graph = nx.DiGraph() if question["is_directed"] else nx.Graph()
    graph.add_nodes_from(range(data["num_nodes"]))
    graph.add_edges_from(data["edges"])
    if question["task"] == "cycle":
        response = "There is a cycle." if nx.cycle_basis(graph) else "There is no cycle."
    else:
        response = "The solution is " + json.dumps(list(nx.topological_sort(graph))) + "."
    score = evaluate_answer(question["task"], response, data, question["is_weighted"],
                            question["is_directed"], question["query_params"])
    if score != 1:
        raise ValueError(f"Upstream grader rejected {question['task']}/{question['graph_id']}")


def edge_set(edges, directed):
    return Counter(tuple(edge) if directed else tuple(sorted(edge)) for edge in edges)


def prepare(config_path, output):
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["upstream_commit"]:
        raise ValueError(f"Expected GraphDO {config['upstream_commit']}, found {commit}")
    if not set(config["tasks"]) <= {"cycle", "topology"}:
        raise ValueError("This setup supports cycle and topology")
    if output.exists():
        raise FileExistsError(f"Benchmark already exists: {output}")

    make_prompt, make_questions = upstream_functions()
    rng = random.Random(config["seed"])
    random.seed(config["seed"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".prepare-") as temporary:
        stage = Path(temporary) / "benchmark"
        graph_root = stage / "graphs"
        stage.mkdir()
        generator = GraphDescriptionGenerator(str(graph_root))
        count = 0
        with (stage / "manifest.jsonl").open("w") as manifest, (stage / "build.log").open("w") as log:
            for task in config["tasks"]:
                records = generate_task(task, config["graphs_per_task"], rng)
                save_jsonl(records, str(graph_root / f"{task}.jsonl"))
                with redirect_stdout(log):
                    descriptions = generator.generate_task_descriptions(task, orders=config["orders"])
                questions = make_questions(descriptions)
                if len(questions) != config["graphs_per_task"]:
                    raise ValueError(f"{task}: incorrect question count {len(questions)}")
                examples = get_examples(task)
                for question in questions:
                    validate_answer_path(question)
                    ordered_descriptions = question.pop("descriptions")
                    expected = edge_set(question["graph_data"]["edges"], question["is_directed"])
                    for order in config["orders"]:
                        description = ordered_descriptions[order]
                        if "error" in description:
                            raise ValueError(description["error"])
                        if edge_set(description["ordered_edges"], question["is_directed"]) != expected:
                            raise ValueError(f"{task}/{question['graph_id']}/{order}: changed graph edges")
                        for style in config["prompt_styles"]:
                            case = dict(question, order=order, prompt_style=style)
                            case["case_id"] = f"{task}/{question['question_id']}/{order}/{style}"
                            case["prompt"] = make_prompt(None, description["description"],
                                                         question["question"], style, examples)
                            manifest.write(json.dumps(case, ensure_ascii=False) + "\n")
                            count += 1
        config["prompt_count"] = count
        config["planned_requests"] = count * len(config["models"])
        config["manifest_sha256"] = sha256((stage / "manifest.jsonl").read_bytes()).hexdigest()
        (stage / "provenance.json").write_text(json.dumps(config, indent=2) + "\n")
        stage.rename(output)
    print(f"{count} prompts ready, {config['planned_requests']} model requests planned: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark" / "subset")
    args = parser.parse_args()
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
