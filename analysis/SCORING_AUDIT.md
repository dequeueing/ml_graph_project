# Scoring audit and report refresh

> This records the original local audit. The GitHub export contains one redacted input and omits its unredacted frozen copy. See [PUBLICATION.md](PUBLICATION.md) for export hashes and exact-replay limits. The original local release is unchanged.

## Current scope

The current report uses `audited_final_scores.json` for traditional tasks and
`node_score_audit.json` for node classification. Earlier server scores and the
first local audit are superseded. All source responses are preserved, including
the 5,991 existing GLM repairs. No additional model calls were made.

At the user's instruction, 62 Hamiltonian graphs whose prompts omit isolated
vertices are excluded from every model, order, and prompt style. The main sample
contains 78 Hamiltonian graphs and 140 graph questions for each other task,
giving 15,950 responses per model and 47,850 across three models. The excluded
4,650 responses remain archived. Node classification contains 4,500 responses
across DeepSeek and Luna. GLM's node run remains excluded because 1,572 outputs
are blank.

Task-family means weight all 25 task and prompt-style cells equally. They are
not pooled percentages over the unequal task sample sizes.

## What was corrected

The first audit repaired missed `Topological sort:` answers, paths followed by
weights or explanations, and formatting variants. This second review found
additional errors involving final corrections, reversed routes in undirected
graphs, direct-edge answers, and alternative paths. It distinguishes a final
answer from a valid intermediate candidate. Tool calls without an answer do
not receive credit.

Node classification is now scored from the prediction for the queried node.
Neighbor labels, vertex IDs, abstentions, and tool code are not predictions.
Explicit target predictions can still be scored when the model expresses
uncertainty about them.

The main scoring path has no released-evaluator fallback. Exceptional saved
answers have explicit decisions in `manual_answer_review.json`, with hashes
of their source responses. There are 284 review entries, including
seven ambiguous responses. No retained response is left with an unsupported
extraction result. This does not imply that every natural-language interpretation
has been proved correct.

## Score changes and current results

Relative to the **first local audit on the same retained traditional cases**,
this review recovers 345 correct answers previously scored
zero and removes 1 spurious correct score. These counts
exclude the effect of removing the underspecified graphs and are additional to
the first audit's corrections.

| Model | Retained responses | Recovered correct | Removed credit | Macro accuracy (%) | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | 15,950 | 23 | 1 | 97.27 | 1 |
| GLM-5.3 | 15,950 | 61 | 0 | 95.78 | 5 |
| GPT-6 Luna | 15,950 | 261 | 0 | 87.16 | 1 |

Node scoring recovers 17 missed predictions and removes
69 spurious correct scores. DeepSeek has 1,547 correct
responses out of 2,250 and Luna has 1,151. Their PPR effects relative to random
order are +2.44 and −11.56 percentage points. Tool-only responses account for
381 DeepSeek outputs, while 728 Luna outputs contain no target prediction.

The current traditional-task comparisons comprise 122 improvements, 107 ties,
and 71 declines. None of the 300 exact paired tests is significant after Holm
correction. The minimum adjusted p-value is 0.163833797. GLM's mean effects are
−0.51 points for BFS, +0.06 for DFS, +0.13 for PR, and −0.26 for PPR. The previous
claim that every structured order reduces GLM's mean accuracy is superseded.
Hamiltonian path has the highest mean ordering variance for DeepSeek and Luna,
and topological sorting for GLM.

## Remaining ambiguity and its effect

The following seven responses contain incompatible answers or malformed output.
They receive zero in the main analysis. They are not being described as seven
definitively wrong answers.

| Model | Case ID | Reason |
| --- | --- | --- |
| deepseek-v4.1-flash | `topology/5/random/cot_bag/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |
| gpt-6-luna | `shortest_path/59/pagerank/few_shot/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |
| glm-5.3 | `shortest_path/131/ppr/few_shot/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |
| glm-5.3 | `hamilton/71/dfs/few_shot/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |
| glm-5.3 | `topology/60/random/zero_shot/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |
| glm-5.3 | `topology/87/dfs/cot/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |
| glm-5.3 | `topology/114/bfs/zero_shot_cot/r1` | Response endorses incompatible alternatives or has a malformed final token. Retain zero and report an upper sensitivity bound. |

`check_ambiguity.py` enumerates all 128 binary score assignments for these cases.
No assignment produces a significant traditional-task comparison after Holm
correction. Every model's mean structured-order effect retains its sign. The
largest change in any such mean effect is
0.07985 percentage points.
These bounds concern the seven identified cases, not hypothetical undiscovered
interpretation errors.

## Validation and limits

- The legacy pipeline reproduces all 52,500 pre-audit traditional scores before
  applying corrections. The current pipeline never uses those scores as fallback.
- Graph specifications agree across all order and prompt conditions. Ground truth
  for 700 unique graph/question instances is cross-checked using independent
  algorithms: union-find, Bellman–Ford, Hamiltonian subset dynamic programming,
  and Kahn's topological check.
- Every selected sequence is independently checked for edge validity, coverage,
  endpoint constraints, or minimum weight as appropriate. An incorrect selected
  path remains incorrect even if another candidate in the response is valid.
- The false-negative screen examined the first audit's 5,701 zero scores. Its
  120 graph-valid candidate flags were inspected individually. Binary conclusions,
  Hamiltonian corrections, and 90 remaining unsupported answer formats were also
  reviewed. These are targeted checks, not a claim of manual review of every answer.
- The 284 exceptional-answer decisions are bound to exact response hashes.
  Regression tests cover the observed parser failures and target-aware node labels.
- RQ2 independently cross-checks all 300 RQ1 paired comparisons. The report check
  verifies Tables 7–9 and B1, macro means, exclusions, source hashes, figure paths,
  and the no-semicolon rule.

No known confirmed false negatives remain among the reviewed flags. The checks
substantially strengthen the scoring evidence, but finite parsing rules and
targeted review cannot establish a universal zero-error guarantee. Ambiguous
responses and their sensitivity bounds remain explicit.

## Reproduction

Run from the project root:

```sh
python3 -m unittest discover -s analysis -p test_score_answers.py
python3 analysis/audit_node_scores.py
python3 analysis/review_scores.py
python3 analysis/analyze_rq1.py
python3 analysis/analyze_rq2.py
python3 analysis/report_metrics.py
python3 analysis/check_ambiguity.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_final_accuracy.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_order_consistency.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_shortest_path_ppr.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_node_ppr.py
python3 analysis/document_audit.py
```

`audit_scores.py` is a compatibility entry point for `review_scores.py`.
`document_audit.py` checks the report and refreshes this audit record and the
compact `rq2_final_scores.json.gz` snapshot. It does not rewrite report prose.
Figures 2–5 use `graphdo-reviewed-*.png` and matching SVG files. All four rendered
figures were visually checked after this refresh.

## Evidence files

- `scoring_audit_inputs.json.gz`: frozen traditional responses and graph metadata.
- `verification_prompt_inputs.json.gz`: original prompts and node-classification inputs.
- `audited_final_scores.json`: main `scores`, full `archived_scores`, exclusions, and hashes.
- `manual_answer_review.json`: exceptional-answer decisions, evidence, and response hashes.
- `false_negative_review.json`: retained-case changes, coverage, and ambiguous responses.
- `node_score_audit.json`: target-aware scores, changed responses, and source hashes.
- `rq1_insights.json`, `rq2_insights.json`, and `report_metrics.json`: derived results.
- `ambiguity_sensitivity.json`: exhaustive assignments for the seven ambiguous cases.
- `before-scoring-audit/` and `before-false-negative-review/`: earlier report and scoring states.

`scoring_audit.json`, `false_negative_screen.json`, and `zero_score_catalog.json`
are historical audit-stage evidence, not the current score source. Server files
remain unchanged. Appendix A points to the local snapshots used by the report.
