"""Generate a tiny deterministic RQ1 manifest without calling an LLM API."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
RQ1_DIR = ROOT / "rq1"
sys.path.insert(0, str(RQ1_DIR))

from graphqa_src import graph_generators  # noqa: E402
from graphqa_src import graph_text_encoders  # noqa: E402


ALGORITHMS = ("er", "ba", "sfn", "sbm", "path", "star", "complete")
ENCODERS = {
    "adjacency": graph_text_encoders.adjacency_encoder,
    "incident": graph_text_encoders.incident_encoder,
}


def graph_has_cycle(graph: nx.Graph) -> bool:
    return bool(nx.cycle_basis(graph))


def make_cases(graph, algorithm: str, graph_index: int, seed: int):
    nodes = sorted(graph.nodes())
    first = nodes[0]
    second = nodes[1] if len(nodes) > 1 else nodes[0]
    names = {node: str(node) for node in nodes}

    task_values = {
        "edge_existence": graph.has_edge(first, second),
        "node_degree": int(graph.degree(first)),
        "cycle_detection": graph_has_cycle(graph),
    }
    for encoding_name, encoder in ENCODERS.items():
        encoded = encoder(graph, names)
        for task_name, answer in task_values.items():
            yield {
                "algorithm": algorithm,
                "graph_index": graph_index,
                "seed": seed,
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "task": task_name,
                "query_nodes": [first, second],
                "encoding": encoding_name,
                "graph_text": encoded,
                "answer": answer,
            }


def main() -> None:
    seed = 20261028
    output = ROOT / "rq1" / "manifests" / "smoke_manifest.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for offset, algorithm in enumerate(ALGORITHMS):
        graph_seed = seed + offset
        graphs = graph_generators.generate_graphs(
            number_of_graphs=1,
            algorithm=algorithm,
            directed=False,
            random_seed=graph_seed,
        )
        records.extend(make_cases(graphs[0], algorithm, 0, graph_seed))

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote {len(records)} smoke-test cases to {output}")


if __name__ == "__main__":
    main()
