# Reproduction guide

## Verify the saved artifacts

From the repository root, run:

```sh
python3 tools/verify_artifacts.py
```

This verifies the 39 included files from the frozen scoring manifest, the separately hashed redacted input, the three benchmark-manifest checksums and case counts, the agreement between the main experiment prompts and the local scoring inputs, and the current report's 11 image links. It does not change scores or invoke a model. One original input is intentionally withheld because of a credential-like string in a saved answer. See [publication details](analysis/PUBLICATION.md).

The original acceptance process and its outputs are preserved in `analysis/releases/scoring-v1-20261122/MANIFEST.json`. That acceptance reproduced all retained scores, checked prompt alignment, passed 28 regression tests, and reproduced the four figures available at acceptance. Its report snapshot retains the original workstation paths, so use the current root report for portable image links.

## Work with the existing results

The main sources are:

- `analysis/audited_final_scores.json` for traditional tasks. Use `scores` for the retained sample and `archived_scores` for all originally generated cases.
- `analysis/node_score_audit.json` for DeepSeek and Luna node classification.
- `analysis/scoring_audit_inputs.json.gz` for the traditional responses and graph metadata, including the recovered GLM answers.
- `analysis/verification_prompt_inputs.json.gz` for prompt verification and node-classification responses.
- `analysis/manual_answer_review.json` for the 284 exceptional-answer decisions, with source-text hashes.
- `analysis/rq1_insights.json`, `analysis/rq2_insights.json`, and `analysis/rq3_insights.json` for the derived findings.

Ordinary report editing and figure work should use these saved scores. The seven known ambiguous answers and their sensitivity analysis are documented in `analysis/SCORING_AUDIT.md`. Historical audit files and server statistics are retained for provenance and do not replace the accepted scores.

To regenerate figures, create a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
```

The following commands read the existing results. Independent figure generators can run in parallel.

```sh
.venv/bin/python figures/plot_edge_order_example.py
.venv/bin/python figures/plot_final_accuracy.py
.venv/bin/python figures/plot_order_consistency.py
.venv/bin/python figures/plot_shortest_path_ppr.py
.venv/bin/python figures/plot_node_ppr.py
.venv/bin/python figures/plot_task_sensitivity.py
.venv/bin/python figures/plot_rq2_findings.py --figure ranking
.venv/bin/python figures/plot_rq2_findings.py --figure prompting
.venv/bin/python figures/plot_rq3_findings.py --figure prompt
.venv/bin/python figures/plot_rq3_findings.py --figure node
.venv/bin/python figures/plot_rq3_findings.py --figure control
```

Replaying the scorer is a separate operation. Exact replay of the original accepted input requires the unredacted file retained in the local source project. The GitHub redaction changes its file hash and has not been presented as a new scoring release. If replay is needed, do it in a disposable copy, following `analysis/SCORING_AUDIT.md`. The dependency order is node scoring, traditional review, RQ1, RQ2, then derived metrics and sensitivity checks. RQ3 reads the saved input and score files through `analysis/analyze_rq3.py`. Do not overwrite the frozen release. A scoring-policy change or confirmed correction requires a new documented version.

## Inspect or run the model experiments

The main design is `experiments/half-scale/experiment.json`. It specifies 17,500 traditional prompts per model and 2,250 node-classification prompts per model. Each manifest includes the exact prompt, graph, query, order, and case ID. The accepted analysis later applies the sample exclusions described above.

The runner requires Python 3.12 or later because it uses `itertools.batched`. Install the core upstream dependencies in an environment separate from the analysis environment:

```sh
git submodule update --init --recursive
python3 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -e experiments/GraphDO
experiments/.venv/bin/python experiments/half-scale/run.py \
  --group traditional --model deepseek-v4.1-flash \
  --initial 1 --maximum 1 --dry-run
experiments/.venv/bin/python experiments/half-scale/run.py \
  --group node --model gpt-6-luna \
  --initial 1 --maximum 1 --dry-run
```

Dry runs check the saved manifests and make no requests. Actual runs require the original host integrations or equivalent user-configured endpoints:

- DeepSeek uses the user's `~/.local/bin/claude-ds` launcher with an explicit model and tools disabled.
- GLM reads `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` from the `env` object in the user's `~/.claude/settings.json`.
- Luna uses an authenticated `codex` CLI with an explicit model and the recorded invocation settings.

Credentials are external to this repository. Removing `--dry-run` makes model requests and writes resumable SQLite results under `experiments/half-scale/runs/`. The model runners use POSIX facilities. The batch launch and recovery shell scripts were written for Linux and retain the original server concurrency settings.

The initial GLM adapter caps output at 400 tokens. The historical recovery scripts under `experiments/half-scale/repair-glm-4096/` selected 5,991 truncated or blank traditional answers and retried with larger limits. They depend on the original SQLite databases, which remain on the server. Their recovered answers are already included in the saved scoring inputs. Their old scoring code is superseded by `analysis/score_answers.py`.

## Benchmark generation and provenance

Use the committed manifests for the recorded experiment. The generation scripts refuse to replace an existing benchmark directory. A fresh generation is a new dataset and needs its own provenance and validation.

`experiments/requirements.lock.txt` records the original core server versions. Each main benchmark's `provenance.json` records the package versions actually used for that group. Node generation additionally needs Torch, PyTorch Geometric, and littleballoffur, as well as citation-network downloads. It used the Forest Fire compatibility change documented in `experiments/half-scale/logs/node-sampler-compatibility.json`. The installed package trees and dataset caches are not included.

The source experiment was imported from `/mnt/new_ssd_3/taojie/comp5222-graphdo/` on the project server on 2026-11-22. The report, reviewed scores, audit evidence, and figures were imported from the local COMP5222 project. The upstream GraphDO commit, included frozen files, and original local snapshot are unchanged. The root report's image URLs were converted to relative paths for GitHub. The one input redaction and omitted frozen file are recorded in `analysis/PUBLICATION_MANIFEST.json`.
