"""Enumerate all 128 binary score assignments for the seven ambiguous answers."""
import json,math,itertools,hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean
HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'audited_final_scores.json').read_text());audit=json.loads((HERE/'false_negative_review.json').read_text())
amb=[(r['model'],r['case_id']) for r in audit['ambiguous']];assert len(amb)==7
scores={m:dict(d['scores']) for m,d in s['models'].items()}
pairs=[]
for m,ss in scores.items():
 groups=defaultdict(dict)
 for k,v in ss.items():
  t,g,o,p,r=k.split('/');groups[t,p][g,o,r]=k
 for (t,p),d in groups.items():
  for o in ['bfs','dfs','pagerank','ppr']:
   pairs.append((m,t,p,o,[(k,d[g,'random',r]) for (g,order,r),k in d.items() if order==o]))
min_p=[];sig=[];max_delta=0;effects=defaultdict(list)
base=json.loads((HERE/'rq1_insights.json').read_text())['traditional']
for config in itertools.product([0,1],repeat=7):
 overrides=dict(zip(amb,config));ps=[];deltas=defaultdict(list)
 for m,t,p,o,ks in pairs:
  gain=loss=0
  for a,b in ks:
   av=overrides.get((m,a),scores[m][a]);bv=overrides.get((m,b),scores[m][b]);gain+=av>bv;loss+=av<bv
  n=gain+loss;pv=min(1,2*sum(math.comb(n,i) for i in range(min(gain,loss)+1))/2**n) if n else 1
  ps.append(pv);deltas[m,o].append(100*(gain-loss)/len(ks))
 adjusted=[];prev=0
 for i,pv in enumerate(sorted(ps)):
  prev=max(prev,min(1,pv*(300-i)));adjusted.append(prev)
 min_p.append(min(adjusted));sig.append(sum(p<.05 for p in adjusted))
 for (m,o),vals in deltas.items():
  effects[m,o].append(mean(vals));max_delta=max(max_delta,abs(mean(vals)-base[m]['mean_delta_pp'][o]))
out={'source_sha256':hashlib.sha256((HERE/'audited_final_scores.json').read_bytes()).hexdigest(),'review_sha256':hashlib.sha256((HERE/'false_negative_review.json').read_bytes()).hexdigest(),'ambiguous_cases':amb,'assignments_tested':len(sig),'significant_comparisons_min':min(sig),'significant_comparisons_max':max(sig),'minimum_holm_p_range':[min(min_p),max(min_p)],'maximum_change_in_mean_ordering_effect_pp':max_delta,'mean_ordering_effect_ranges':{m+'/'+o:[min(v),max(v)] for (m,o),v in effects.items()}}
(HERE/'ambiguity_sensitivity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
