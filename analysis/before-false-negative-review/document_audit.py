"""Write audit provenance and verify the report's numeric tables."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
models=[('deepseek-v4.1-flash','DeepSeek V4.1 Flash'),('glm-5.3','GLM-5.3'),('gpt-6-luna','GPT-6 Luna')]
tasks=[('connectivity','Connectivity'),('cycle','Cycle detection'),('shortest_path','Weighted shortest path'),('hamilton','Hamiltonian path'),('topology','Topological sorting')]
orders=['random','bfs','dfs','pagerank','ppr']
snapshot=json.loads((HERE/'audited_final_scores.json').read_text())
audit=json.loads((HERE/'scoring_audit.json').read_text())
rq1=json.loads((HERE/'rq1_insights.json').read_text())
rq2=json.loads((HERE/'rq2_insights.json').read_text())
report=(ROOT/'COMP5222_GraphDO_Report.md').read_text()
assert snapshot['scorer_sha256']==hashlib.sha256((HERE/'score_answers.py').read_bytes()).hexdigest()
assert rq1['source_sha256']==hashlib.sha256((HERE/'audited_final_scores.json').read_bytes()).hexdigest()
assert ';' not in report and '；' not in report
for m,name in models:
 cells=snapshot['models'][m]['cells']
 assert f"{sum(c['score'] for c in cells)/175:.2f}%" in report
 base=sum(c['score'] for c in cells if c['order']=='random')/35
 v=rq1['traditional'][m]
 deltas=[f'{v["mean_delta_pp"][o]:+.2f}'.replace('-','−') for o in orders[1:]]
 counts=' / '.join(str(v['higher_equal_lower'][o]) for o in ['higher','equal','lower'])
 line='| Traditional | '+name+' | '+f'{base:.2f} | '+' | '.join(deltas)+' | '+counts+' |'
 assert line in report, line
 for t,label in tasks:
  counts=[sum(c['score'] for c in cells if c['task']==t and c['order']==o) for o in orders]
  vals=[f'**{x/7:.2f}**' if x==max(counts) else f'{x/7:.2f}' for x in counts]
  line='| '+label+' | '+name+' | '+' | '.join(vals)+' |'
  assert line in report, line
for t,label in tasks:
 vals=[next(r['mean_variance_pp2'] for r in rq2['task_summary'] if r['task']==t and r['model']==m) for m,_ in models]
 line='| '+label+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |'
 assert line in report, line
old=(HERE/'before-scoring-audit/COMP5222_GraphDO_Report.md').read_text()
assert report[report.index('## Appendix B.'):]==old[old.index('## Appendix B.'):]
assert rq1['node']==json.loads((HERE/'before-scoring-audit/rq1_insights.json').read_text())['node']

# Keep the old compact filename current for any downstream consumers.
compact={k:v for k,v in snapshot.items() if k!='models'}
compact['score_snapshot']='analysis/audited_final_scores.json'
compact['models']={m:{'scores':d['scores']} for m,d in snapshot['models'].items()}
(HERE/'rq2_final_scores.json.gz').write_bytes(gzip.compress(json.dumps(compact).encode(),mtime=0))

rows=[]
for m,label in models:
 c=audit['models'][m]['counts']
 rows.append(f"| {label} | {c['previous']/175:.2f} | {c['score']/175:.2f} | {c['gained']:,} | {c['lost']:,} | {c['verified']:,} | {c['fallback']:,} |")
changed=len(audit['changes'])
gains=sum(c['score']>c['previous'] for c in audit['changes'])
losses=changed-gains
text='''# Scoring audit and report refresh

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

'''+f'''{changed:,} of 52,500 scores changed. {gains:,} changed from incorrect to correct,
and {losses:,} changed from correct to incorrect. All four removed correct scores
were topological answers that omitted vertices required by edges in the prompt.
Node-classification scores are unchanged.

| Model | Previous accuracy (%) | Audited accuracy (%) | Recovered correct | Removed correct | Graph-checked answers | Fallback answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

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
'''
(HERE/'SCORING_AUDIT.md').write_text(text)
print('Verified Tables 7–9, model means, node data, Appendix B, source hashes, and punctuation.')
print('Wrote SCORING_AUDIT.md and refreshed the compact score snapshot.')
