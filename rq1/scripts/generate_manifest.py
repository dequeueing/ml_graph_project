"""Generate a frozen, paper-scale RQ1 case manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rq1"))

from graphqa_src import graph_generators, graph_text_encoders  # noqa: E402


def cycle_exists(graph: nx.Graph) -> bool:
    return bool(nx.cycle_basis(graph))


def choose_edge_pair(graph: nx.Graph, graph_index: int) -> tuple[int, int]:
    nodes = sorted(graph.nodes())
    edges = sorted(tuple(sorted(edge)) for edge in graph.edges())
    non_edges = [
        (nodes[i], nodes[j])
        for i, left in enumerate(nodes)
        for right in nodes[i + 1 :]
        for j in [nodes.index(right)]
        if not graph.has_edge(left, right)
    ]
    candidates = edges if graph_index % 2 == 0 else non_edges
    if not candidates:
        candidates = non_edges or edges
    if not candidates:
        return nodes[0], nodes[0]
    return candidates[graph_index % len(candidates)]


def make_records(graph, algorithm: str, graph_index: int, seed: int, encoding: str):
    nodes = sorted(graph.nodes())
    names = {node: str(node) for node in nodes}
    graph_text = graph_text_encoders.adjacency_encoder(graph, names)
    edge_pair = choose_edge_pair(graph, graph_index)
    degree_node = nodes[graph_index % len(nodes)]
    tasks = [
        ("edge_existence", list(edge_pair), graph.has_edge(*edge_pair)),
        ("node_degree", [degree_node], int(graph.degree(degree_node))),
        ("cycle_detection", [], cycle_exists(graph)),
    ]
    graph_id = f"{algorithm}-{seed}"
    for task, query_nodes, answer in tasks:
        yield {
            "case_id": f"{graph_id}-{task}",
            "graph_id": graph_id,
            "algorithm": algorithm,
            "graph_index": graph_index,
            "seed": seed,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "task": task,
            "query_nodes": query_nodes,
            "encoding": encoding,
            "graph_text": graph_text,
            "answer": answer,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rq1/config/rq1_full.json")
    parser.add_argument("--limit-per-algorithm", type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)

    graph_counter = 0
    total_cases = 0
    with output.open("w", encoding="utf-8") as handle:
        for algorithm, configured_count in config["graphs_per_algorithm"].items():
            count = configured_count
            if args.limit_per_algorithm is not None:
                count = min(count, args.limit_per_algorithm)
            for local_index in range(count):
                seed = int(config["seed"]) + graph_counter
                graph = graph_generators.generate_graphs(
                    number_of_graphs=1,
                    algorithm=algorithm,
                    directed=False,
                    random_seed=seed,
                )[0]
                records = make_records(
                    graph, algorithm, local_index, seed, config["encoding"]
                )
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    total_cases += 1
                graph_counter += 1

    print(f"Wrote {total_cases} cases for {graph_counter} graphs to {output}")


if __name__ == "__main__":
    main()
