# RQ1: Model Capability

## Research question

**Have modern large language models improved their performance on basic graph reasoning tasks compared with the historical GraphQA results?**

## Objective

RQ1 is the reproduction component of the project. It evaluates a small set of modern, API-accessible LLMs on the core GraphQA tasks and compares the resulting performance with the qualitative and quantitative findings reported for the PaLM-era models. The comparison is intended as contextual evidence rather than a strict measurement of progress across unrelated model families and API conditions.

## Scope

The initial evaluation will use three tasks:

- **Edge existence:** determine whether an edge connects two specified nodes.
- **Node degree:** return the number of neighbors of a specified node.
- **Cycle detection:** determine whether the graph contains at least one cycle.

To isolate model capability, the first RQ1 experiment will keep the graph encoding, prompt template, answer format, graph distribution, and decoding settings fixed across models. The primary encoding will be selected before the final evaluation and recorded in the experiment configuration. Encoding comparisons belong mainly to RQ2.

The initial model set is:

- GLM-5.3 through the configured local API;
- DeepSeek V4.1 Flash through the configured local API; and
- GPT-6 Luna through the configured OpenAI-compatible API.

The exact model identifiers, endpoint settings, access date, and provider-specific limitations will be recorded with every run.

## Experimental design

RQ1 will use paired evaluation. A shared manifest will define the graph instances, task questions, prompts, canonical answers, and random seeds before any model requests are made. Every model will receive the same cases in the same experimental condition. The evaluation split will use held-out graph seeds and will not be used to tune prompts after results are inspected.

The public repository provides the GraphQA generation logic, task definitions, and text encoders, but it does not include a complete pre-generated case archive. RQ1 will therefore reuse the implementation and generate the cases once with fixed seeds. The resulting manifest will be frozen and reused for every model; we will not generate a separate dataset for each model. The existing small API trial is useful for checking connectivity and parsing, but it will be treated as a pilot rather than as the final RQ1 benchmark.

The relevant GraphQA source files are copied into `graphqa_src/` so that the RQ1 experiment can be run from this directory and tracked independently. The copied modules provide graph generation, task definitions, graph metrics, and text encoders; they do not contain a separate dataset.

The core comparison will use zero-shot prompts with an explicit answer format. Binary tasks will request `Yes` or `No`, and node-degree questions will request an integer. Only the normalized final answer will be scored in the primary analysis. Explanations may be retained for error inspection but will not affect correctness.

## Measurements

The primary measure is exact-match accuracy, reported separately for each model and task. We will also report invalid-response rates and per-task error patterns. Latency, token usage, request failures, and available cost information will be recorded as supplementary measurements so that differences in model capability can be considered alongside practical API behavior.

Results will be summarized both in aggregate and by graph family. Error analysis will distinguish reasoning errors from formatting errors, parsing failures, and API failures. Historical GraphQA results will be used as reference points, with differences interpreted cautiously when the original model, graph sample, or service conditions are not directly reproducible.

## Planned artifacts

The RQ1 directory will eventually contain:

```text
rq1/
├── README.md
├── graphqa_src/              # copied GraphQA generation and encoding modules
├── config/                   # frozen experiment settings
├── scripts/                  # manifest generation and evaluation scripts
├── prompts/                  # frozen model prompts
├── manifests/
├── results/
├── logs/
└── figures/
```

The central data artifact will be a frozen case manifest containing graph identifiers, serialized graphs, tasks, questions, encodings, and ground-truth answers. All generated cases, prompts, model responses, normalized predictions, and summary tables should be traceable to this manifest and one versioned configuration. Credentials and provider secrets must remain outside the repository.

## Initial validation plan

We will validate the pipeline in progressively larger steps:

1. Run `talk-like-a-graph/.venv311/bin/python rq1/scripts/smoke_test.py` (or an equivalent environment with NetworkX and NumPy) to generate one graph for each GraphQA topology family. This creates a small manifest with seven graphs, two encodings, and three tasks, and does not call an LLM API.
2. Inspect graph metadata, encoded prompts, and ground-truth answers manually and with a lightweight parser check.
3. Run a small API pilot on a limited subset of the smoke manifest to verify endpoint configuration, answer normalization, logging, and retry handling.
4. Freeze the RQ1 configuration and manifest before any larger evaluation.

No large-scale model run will be started during this initial validation stage.

## One-case end-to-end validation

After the smoke manifest has been generated, one complete model call can be run with:

```bash
python3 rq1/scripts/run_one_case.py --profile codex --index 0
```

The script reads one manifest record, builds the task prompt, calls the selected local API, extracts the answer, compares it with the stored ground truth, and writes the complete audit record to `results/one_case_<profile>_index_<index>.json`. The first validation used `codex` with the configured `gpt-6-luna` model on an edge-existence case. The API returned `Yes`, the parser produced `true`, and the stored ground truth was `true`, so the case was judged correct. This confirms the graph-to-prompt, API, parsing, and scoring path before any larger run.

A seven-case topology pilot was then run with the same profile, task, and adjacency encoding. It used one case from each of ER, BA, SFN, SBM, path, star, and complete graphs. All seven responses were valid and matched their ground-truth labels. The records are stored as `results/one_case_codex_index_<index>.json`; this pilot is still an infrastructure check, not a benchmark result.

## Status

The smoke and one-case validation stages are complete. The next step is to freeze the small pilot configuration and decide the formal RQ1 sample size before running additional model calls.
