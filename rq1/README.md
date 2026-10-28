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

RQ1 will reuse the existing GraphQA generation logic, task definitions, and any validated benchmark cases already available in the workspace. We will not generate a separate dataset for each model. If a complete serialized case set is not available, we will generate the cases once with fixed seeds, save the resulting manifest, and reuse that frozen manifest for every model. The existing small API trial is useful for checking connectivity and parsing, but it will be treated as a pilot rather than as the final RQ1 benchmark.

The core comparison will use zero-shot prompts with an explicit answer format. Binary tasks will request `Yes` or `No`, and node-degree questions will request an integer. Only the normalized final answer will be scored in the primary analysis. Explanations may be retained for error inspection but will not affect correctness.

## Measurements

The primary measure is exact-match accuracy, reported separately for each model and task. We will also report invalid-response rates and per-task error patterns. Latency, token usage, request failures, and available cost information will be recorded as supplementary measurements so that differences in model capability can be considered alongside practical API behavior.

Results will be summarized both in aggregate and by graph family. Error analysis will distinguish reasoning errors from formatting errors, parsing failures, and API failures. Historical GraphQA results will be used as reference points, with differences interpreted cautiously when the original model, graph sample, or service conditions are not directly reproducible.

## Planned artifacts

The RQ1 directory will eventually contain:

```text
rq1/
├── README.md
├── config/
├── prompts/
├── manifests/
├── results/
├── logs/
└── figures/
```

The central data artifact will be a frozen case manifest containing graph identifiers, serialized graphs, tasks, questions, encodings, and ground-truth answers. All generated cases, prompts, model responses, normalized predictions, and summary tables should be traceable to this manifest and one versioned configuration. Credentials and provider secrets must remain outside the repository.

## Status

This directory currently contains the experiment specification only. The next step is to identify the reusable GraphQA cases, finalize the fixed encoding, generate or freeze the shared manifest if necessary, and validate the answer parser before running the model APIs.
