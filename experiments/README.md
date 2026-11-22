# COMP5222 GraphDO replication

> This is the original two-task pilot README, preserved with the experiment source. The final experiment is in `half-scale/`. Use the [repository README](../README.md) and [reproduction guide](../REPRODUCTION.md) for the current scope and accepted scoring results. The provisional evaluator discussed below was superseded by the reviewed local analysis.

## Scope

The current configuration covers cycle detection and topological sorting, with 150 graphs per task, random/BFS/DFS edge orders, one official few-shot prompt style, and three models. This produces 900 prompts and 2,700 planned requests. RQ1 compares order accuracy against random, RQ2 compares the variance across selected orders, and RQ3 compares the selected order rankings by task and model. The original paper also evaluates other tasks, orders, and prompting styles.

The GraphDO checkout is pinned at 4ba8f925c76574e2fca8d4f32f753b57a341b44f. The benchmark uses the official generator, encoder, edge ordering, questions, examples, prompt construction, and evaluator. The two pure question and prompt functions are loaded from their pinned sources so the CUDA inference dependencies are unnecessary. No templates or scoring rules were rewritten. The manifest contains the original constructed prompt and graph for each case. Preparation checks that all three orders preserve the input edges and the official grader accepts a known-correct response for every graph.

The official cycle evaluator recognizes specific phrases such as “there is a cycle.” In the initial smoke test it scored some affirmative Yes answers as zero. Scoring follows the pinned evaluator, and raw answers are saved for inspection.

## Model settings

| Model | Interface | Settings |
| --- | --- | --- |
| deepseek-v4.1-flash | Existing claude-ds launcher | Explicit model, tools disabled, no persisted session |
| glm-5.3 | Existing Anthropic-compatible gateway | Temperature 0, max 400 output tokens, sends thinking.type=disabled |
| gpt-6-luna | Codex CLI | Explicit model, low reasoning effort, shell and web search disabled, read-only ephemeral session |

The application prompt is identical across the three routes for each case. CLI wrappers can add system context and do not expose the same decoding controls as the direct API. Returned model IDs are checked when available. The adapters read the server's existing credentials at runtime.

## Commands

From this directory, use the project Python at .venv/bin/python. requirements.lock.txt records the installed versions. benchmark/subset/provenance.json records the benchmark settings and manifest checksum.

    .venv/bin/python run_experiment.py run --dry-run
    .venv/bin/python run_experiment.py run --graphs-per-task 1 --orders random --db runs/smoke-verified.sqlite3
    .venv/bin/python run_experiment.py run --db runs/formal.sqlite3
    .venv/bin/python run_experiment.py summary --db runs/formal.sqlite3 --output runs/summary.json

The formal run makes billable model requests. Reusing its database resumes failed or unstarted cases and skips successful records. A single process writes to each database. Response text, model IDs when emitted, timing, usage, and errors are stored locally. API failures are tracked separately from wrong answers. The summary publishes accuracy after all planned cases in the cell have succeeded.

To change the selected prompt styles or orders, copy experiment.json to a new configuration, edit its arrays and then use a fresh benchmark directory and results database. Supported GraphDO prompt styles are zero_shot, zero_shot_cot, few_shot, cot, and cot_bag. Supported orders are random, bfs, dfs, pagerank, and ppr.

    .venv/bin/python prepare_benchmark.py --config experiment.json --output benchmark/another-run
    .venv/bin/python run_experiment.py --benchmark benchmark/another-run run --dry-run
