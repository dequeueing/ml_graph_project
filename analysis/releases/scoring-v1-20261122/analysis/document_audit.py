"""Check current report tables and provenance, then write the audit record."""
from pathlib import Path
from collections import Counter
from statistics import mean
import gzip
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = [('deepseek-v4.1-flash', 'DeepSeek V4.1 Flash'),
          ('glm-5.3', 'GLM-5.3'), ('gpt-6-luna', 'GPT-6 Luna')]
TASKS = [('connectivity', 'Connectivity'), ('cycle', 'Cycle detection'),
         ('shortest_path', 'Weighted shortest path'), ('hamilton', 'Hamiltonian path'),
         ('topology', 'Topological sorting')]
ORDERS = ['random', 'bfs', 'dfs', 'pagerank', 'ppr']


def read(name):
    return json.loads((HERE / name).read_text())


def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def signed(value):
    return f'{value:+.2f}'.replace('-', '−')


def maxima(values):
    return [f'**{v:.2f}**' if abs(v - max(values)) < 1e-10 else f'{v:.2f}'
            for v in values]


snapshot = read('audited_final_scores.json')
audit = read('false_negative_review.json')
node = read('node_score_audit.json')
reviews = read('manual_answer_review.json')
rq1 = read('rq1_insights.json')
rq2 = read('rq2_insights.json')
metrics = read('report_metrics.json')
sensitivity = read('ambiguity_sensitivity.json')
report = (ROOT / 'COMP5222_GraphDO_Report.md').read_text()

for field, filename in [('source_sha256', 'scoring_audit_inputs.json.gz'),
                        ('scorer_sha256', 'score_answers.py'),
                        ('review_sha256', 'manual_answer_review.json'),
                        ('review_script_sha256', 'review_scores.py')]:
    assert snapshot[field] == digest(filename), filename
assert node['source_sha256'] == digest('verification_prompt_inputs.json.gz')
assert node['scorer_sha256'] == digest('audit_node_scores.py')
assert rq1['source_sha256'] == digest('audited_final_scores.json')
assert rq1['node_source_sha256'] == digest('node_score_audit.json')
assert rq2['provenance']['score_snapshot_sha256'] == digest('audited_final_scores.json')
assert metrics['source_sha256'] == digest('audited_final_scores.json')
assert metrics['rq1_sha256'] == digest('rq1_insights.json')
assert metrics['rq2_sha256'] == digest('rq2_insights.json')
assert sensitivity['source_sha256'] == digest('audited_final_scores.json')
assert sensitivity['review_sha256'] == digest('false_negative_review.json')
assert ';' not in report and '；' not in report

excluded = set(snapshot['excluded_hamilton_graph_ids'])
assert len(excluded) == 62 and excluded == set(audit['excluded_hamilton_graph_ids'])
assert audit['independent_graph_checks'] == 700
assert audit['unextracted'] == node['unextracted'] == []
assert len(audit['ambiguous']) == 7
common_keys = None
for model, name in MODELS:
    data = snapshot['models'][model]
    scores = dict(data['scores'])
    archived = dict(data['archived_scores'])
    assert len(scores) == len(data['scores']) == 15950
    assert len(archived) == len(data['archived_scores']) == 17500
    assert set(scores) == {k for k in archived
                          if not (k.startswith('hamilton/') and k.split('/')[1] in excluded)}
    assert all(archived[k] == v for k, v in scores.items())
    if common_keys is None:
        common_keys = set(scores)
    assert set(scores) == common_keys
    cells = data['cells']
    assert len(cells) == 125
    assert all(c['n'] == (78 if c['task'] == 'hamilton' else 140) for c in cells)
    for c in cells:
        selected = [v for k, v in scores.items()
                    if (k.split('/')[0], k.split('/')[2], k.split('/')[3])
                    == (c['task'], c['order'], c['prompt'])]
        assert len(selected) == c['n'] and sum(selected) == c['score']
    means = metrics['models'][model]
    assert abs(means['mean_accuracy'] - mean(100*c['score']/c['n'] for c in cells)) < 1e-10
    assert f"{means['mean_accuracy']:.2f}%" in report.split('## 1. Introduction')[0]
    v = rq1['traditional'][model]
    counts = ' / '.join(str(v['higher_equal_lower'].get(x, 0)) for x in ['higher', 'equal', 'lower'])
    line = ('| Traditional | ' + name + f" | {means['order_means']['random']:.2f} | "
            + ' | '.join(signed(v['mean_delta_pp'][o]) for o in ORDERS[1:]) + ' | ' + counts + ' |')
    assert line in report, line
    for task, label in TASKS:
        values = [means['tasks'][task][o] for o in ORDERS]
        line = '| ' + label + ' | ' + name + ' | ' + ' | '.join(maxima(values)) + ' |'
        assert line in report, line
    if model in node['models']:
        n = means['node']
        counts = ' / '.join(str(n['outcomes'][x]) for x in ['higher', 'equal', 'lower'])
        line = (f"| Node classification | {name} | {n['random']:.2f} | "
                + ' | '.join(signed(n['deltas'][o]) for o in ORDERS[1:]) + ' | ' + counts + ' |')
        assert line in report, line
        nd = node['models'][model]
        assert len(nd['scores']) == nd['counts']['n'] == 2250
        cells = {(r['group'], r['order']): r for r in nd['cells']}
        for dataset, label in [('cora', 'Cora'), ('citeseer', 'CiteSeer'), ('pubmed', 'PubMed')]:
            for sampler, sampler_label in [('ego', 'Ego'), ('forest_fire', 'Forest Fire')]:
                records = [cells[dataset+'/'+sampler, o] for o in ORDERS]
                assert all(r['n'] == 75 for r in records)
                values = [100*r['correct']/r['n'] for r in records]
                line = f'| {label} | {sampler_label} | {name} | ' + ' | '.join(maxima(values)) + ' |'
                assert line in report, line
for task, label in TASKS:
    values = [next(r['mean_variance_pp2'] for r in rq2['task_summary']
                   if r['task'] == task and r['model'] == m) for m, _ in MODELS]
    line = '| ' + label + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |'
    assert line in report, line

assert len(rq1['paired_comparisons']) == 300 and len(rq2['cells']) == 75
assert sensitivity['assignments_tested'] == 128
assert sensitivity['significant_comparisons_max'] == 0
assert sensitivity['maximum_change_in_mean_ordering_effect_pp'] < .08
assert all(lo*hi > 0 for lo, hi in sensitivity['mean_ordering_effect_ranges'].values())
for path in re.findall(r'!\[[^\]]*\]\(<([^>]+)>\)', report):
    assert Path(path).is_file(), path
for suffix in ['accuracy', 'consistency', 'shortest-path-ppr', 'ppr-breakdown']:
    stem = f'graphdo-reviewed-{suffix}'
    assert stem + '.png' in report
    assert all((ROOT / 'figures' / (stem + ext)).is_file() for ext in ['.png', '.svg'])

compact = {k: v for k, v in snapshot.items() if k != 'models'}
compact.update(score_snapshot='analysis/audited_final_scores.json',
               score_snapshot_sha256=digest('audited_final_scores.json'),
               models={m: {'scores': d['scores']} for m, d in snapshot['models'].items()})
(HERE / 'rq2_final_scores.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(), mtime=0))

rows = []
for model, name in MODELS:
    c = audit['models'][model]
    rows.append(f"| {name} | {c['n']:,} | {c['gained']} | {c['lost']} | "
                f"{metrics['models'][model]['mean_accuracy']:.2f} | {c['ambiguous']} |")
trad_changes = Counter('recovered' if r['score'] > r['previous'] else 'removed' for r in audit['changes'])
node_changes = Counter('recovered' if r['score'] > r['previous'] else 'removed' for r in node['changes'])
assert trad_changes == {'recovered': 345, 'removed': 1}
assert node_changes == {'recovered': 17, 'removed': 69}
ambiguous_rows = '\n'.join(f"| {r['model']} | `{r['case_id']}` | {r['reason'].replace('|', '/')} |"
                           for r in reviews if r['status'] == 'ambiguous')
text = f'''# Scoring audit and report refresh

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
of their source responses. There are {len(reviews)} review entries, including
seven ambiguous responses. No retained response is left with an unsupported
extraction result. This does not imply that every natural-language interpretation
has been proved correct.

## Score changes and current results

Relative to the **first local audit on the same retained traditional cases**,
this review recovers {trad_changes['recovered']} correct answers previously scored
zero and removes {trad_changes['removed']} spurious correct score. These counts
exclude the effect of removing the underspecified graphs and are additional to
the first audit's corrections.

| Model | Retained responses | Recovered correct | Removed credit | Macro accuracy (%) | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Node scoring recovers {node_changes['recovered']} missed predictions and removes
{node_changes['removed']} spurious correct scores. DeepSeek has 1,547 correct
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
{ambiguous_rows}

`check_ambiguity.py` enumerates all 128 binary score assignments for these cases.
No assignment produces a significant traditional-task comparison after Holm
correction. Every model's mean structured-order effect retains its sign. The
largest change in any such mean effect is
{sensitivity['maximum_change_in_mean_ordering_effect_pp']:.5f} percentage points.
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
'''
(HERE / 'SCORING_AUDIT.md').write_text(text)
print('Verified Tables 7–9 and B1, macro means, exclusions, source hashes, figure paths, and punctuation.')
print('Wrote SCORING_AUDIT.md and refreshed rq2_final_scores.json.gz.')
