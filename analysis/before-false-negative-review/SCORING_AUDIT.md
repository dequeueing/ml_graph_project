# Scoring audit and report refresh

The current report uses `audited_final_scores.json`. The earlier server snapshot
`scratch/glm-final-report-20261122.json` is superseded for report analysis.
The saved questions, graph instances, edge lists, visible responses, and existing
GLM repairs have not changed. No additional model requests were made.

## Corrections

The earlier answer extractor only inspected the last line and recognized
`sorting` but not `sort`. It also rejected labeled paths followed by a weight,
repeated paths, and conclusions followed by explanatory text. In some answers it
mistook a final distance, an alternative route, or a partial edge check for the
returned path. These failures particularly affected topological sorting and
weighted shortest path, but the audit covered all five traditional tasks.

The revised extractor selects an answer using its wording and position, then
validates the selected answer against graph truth. It handles common Markdown
and LaTeX formatting. It does not select a candidate because that candidate
passes validation. A declared answer takes precedence over an explanatory
fragment. Negative Hamiltonian answers are checked by exact search.
The original rules for omitted isolated vertices in topological sorting are
unchanged. The released evaluator remains the fallback for unextracted answers.

## Impact

8,613 of 52,500 scores changed. 8,609 changed from incorrect to correct,
and 4 changed from correct to incorrect. All four removed correct scores
were topological answers that omitted vertices required by edges in the prompt.
Node-classification scores are unchanged.

| Model | Previous accuracy (%) | Audited accuracy (%) | Recovered correct | Removed correct | Graph-checked answers | Fallback answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | 72.34 | 92.79 | 3,580 | 0 | 17,119 | 381 |
| GLM-5.3 | 68.93 | 90.86 | 3,838 | 1 | 17,428 | 72 |
| GPT-6 Luna | 76.98 | 83.77 | 1,191 | 3 | 17,359 | 141 |

The remaining fallback counts are not a count of model errors. Some contain
valid answers in unsupported formats, while others contain no usable answer.
For DeepSeek, 365 fallback responses contain serialized tool calls. The audited
score therefore remains subject to the explicit extraction limitation in §6.

The average ordering gains remain small. GLM remains below random order under
all four structured orders. The RQ1 comparison counts are now 106 higher,
99 equal, and 95 lower. No comparison is significant after Holm correction,
and the minimum adjusted p-value is 0.5634676432. Luna's mean traditional-task
PPR gain is now 2.49 percentage points.

The previous claim that shortest path is most sensitive for every model is no
longer supported. Hamiltonian path has the highest mean variance for DeepSeek
and Luna. Shortest path is highest for GLM. Connectivity and cycle detection
remain below all three sequence tasks in every model–prompt combination.

## Verification and reproduction

The audit reproduces all 52,500 previous scores before applying the new extractor.
This checks that the graph validators and fallback inputs match the previous
pipeline. Nineteen regression tests cover observed formats, incorrect answers,
answer precedence, candidate rejection, explanatory edge lists, mathematical
formatting, and negative Hamiltonian decisions. Changed-score examples were
inspected across models and tasks. All four final correct-to-incorrect changes
were checked against their missing edge endpoints.

Run from the project root:

```sh
python3 -m unittest discover -s analysis -p test_score_answers.py
python3 analysis/audit_scores.py
python3 analysis/analyze_rq1.py
python3 analysis/analyze_rq2.py
python3 analysis/document_audit.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_final_accuracy.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_order_consistency.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_shortest_path_ppr.py
MPLCONFIGDIR=/tmp/comp5222-matplotlib python3 figures/plot_node_ppr.py
```

`document_audit.py` verifies Tables 7–9 against the derived counts, checks the
reported model means and the no-semicolon rule, and confirms that Appendix B
and all node-classification records are unchanged. It does not rewrite report
prose. The 300 paired RQ1 tests are independently cross-checked during RQ2
analysis. Figures 2–5 use new `graphdo-audited-*` filenames to avoid cached
previews and have been visually inspected.

## Files

- `scoring_audit_inputs.json.gz`: frozen visible responses, graph metadata,
  released fallback scores, and previous scores for all 52,500 cases.
- `score_answers.py`: current extraction and graph validation.
- `audit_scores.py`: full re-scoring and legacy reproduction assertions.
- `audited_final_scores.json`: current per-case scores, cell totals, and hashes.
- `scoring_audit.json`: every changed case with old and new scores, extracted
  answers, and final response lines, plus coverage and fallback summaries.
- `rq1_insights.json`: all 300 paired tests and unchanged node-classification
  counts, with source hashes.
- `rq2_insights.json`: all 75 model–task–prompt variances, task rankings, prompt
  contrasts, and provenance.
- `rq2_final_scores.json.gz`: compact copy of the current audited scores.
- `before-scoring-audit/`: earlier report, scores, analyses, project memory,
  and figure scripts, retained for comparison.
- `rq2_sort_label_audit.json` and `audit_rq2_extraction.py`: preliminary diagnostic
  of the `sort` label alone. These are historical evidence, not current results.

The server files were read only during this audit. Appendix A points to the
local reproducible snapshot used for the report.
