# ml_graph_project

This repository contains experiments and supporting code for studying how
large language models reason over graph-structured data.

## Contents

- `paper_2310.04560.pdf` and `paper_2310.04560.txt`: the paper *Talk Like a Graph*.
- `talk-like-a-graph/`: the accompanying graph generators, graph tasks, text
  encoders, tests, and tutorial notebooks.
- `query_llms.py`: a dependency-free CLI for querying the locally configured
  GLM, DeepSeek, and Codex-compatible gateways. Credentials are read from the
  existing local configuration and are never stored in this repository.
- `proposal/`: the original `proposal.md`, English outline, current submission draft,
  LaTeX source, and compiled PDF.
- `REPRODUCTION.md`: what can be reproduced locally, setup commands, and the
  limitations of reproducing the original PaLM experiments.

## Running the graph code

The graph package requires Python 3.10 or newer, NetworkX, NumPy, and
`absl-py` for the lightweight graph generation and encoding code. For example:

```bash
cd talk-like-a-graph
python3.11 -m venv .venv
.venv/bin/pip install networkx numpy absl-py
PYTHONPATH=. .venv/bin/python -m talk_like_a_graph.graph_generators_runner \
  --algorithm path --number_of_graphs 2 \
  --output_path /tmp/tlag-run --split validation
```

The full task-generation pipeline additionally depends on TensorFlow,
TensorFlow-GNN, and SeqIO.
