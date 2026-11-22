"""Collect report numbers from the current main-analysis score snapshot."""
import json,gzip,hashlib
from pathlib import Path
from statistics import mean
HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'audited_final_scores.json').read_text());rq1=json.loads((HERE/'rq1_insights.json').read_text());rq2=json.loads((HERE/'rq2_insights.json').read_text());inputs=json.loads(gzip.decompress((HERE/'verification_prompt_inputs.json.gz').read_bytes()))
orders=['random','bfs','dfs','pagerank','ppr'];out={'models':{},'source_sha256':hashlib.sha256((HERE/'audited_final_scores.json').read_bytes()).hexdigest(),'rq1_sha256':hashlib.sha256((HERE/'rq1_insights.json').read_bytes()).hexdigest(),'rq2_sha256':hashlib.sha256((HERE/'rq2_insights.json').read_bytes()).hexdigest()}
for m,d in s['models'].items():
 cells=d['cells'];v={o:mean(100*c['score']/c['n'] for c in cells if c['order']==o) for o in orders}
 tasks={t:{o:mean(100*c['score']/c['n'] for c in cells if c['task']==t and c['order']==o) for o in orders} for t in {c['task'] for c in cells}}
 scores=dict(d['scores']);pairs=[(k,k.replace('/pagerank/','/ppr/')) for k in scores if k.startswith('hamilton/') and '/pagerank/' in k]
 assert len(pairs)==390 and all(inputs['traditional_prompts'][a]==inputs['traditional_prompts'][b] for a,b in pairs)
 out['models'][m]={'mean_accuracy':mean(v.values()),'order_means':v,'tasks':tasks,'identical_hamilton_pairs':len(pairs),'identical_prompt_disagreements':sum(scores[a]!=scores[b] for a,b in pairs),'ppr_shortest':{r['prompt']:r['delta_pp'] for r in rq1['traditional'][m]['ppr_shortest_path_by_prompt']}}
 if m in rq1['node']:
  n=rq1['node'][m];rows=[r for rs in n.values() for r in rs]
  out['models'][m]['node']={'random':mean(100*r['baseline_correct']/r['n'] for r in n['bfs']), 'deltas':{o:mean(r['delta_pp'] for r in n[o]) for o in n},'outcomes':{k:sum((r['delta_pp']>0 if k=='higher' else r['delta_pp']<0 if k=='lower' else r['delta_pp']==0) for r in rows) for k in ['higher','equal','lower']}}
out['rq2_rankings']=rq2['rankings'];out['rq2_prompt_contrasts']=[{k:v for k,v in r.items() if k!='comparisons'} for r in rq2['prompt_contrasts']]
(HERE/'report_metrics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['models'],indent=2))
