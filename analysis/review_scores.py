"""Apply graph-validated answer review and the authorized prompt exclusion.
Run after audit_node_scores.py. No model calls are made.
"""
import gzip,hashlib,json,math
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from functools import lru_cache
from score_answers import adjudicate,validate_path,graph_for,reachable,has_cycle,hamiltonian,minimum_weight
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@lru_cache(None)
def independent_truth(task,n,edges,source=0,target=0):
 # Union-find for connectivity/cycle, Bellman-Ford for distance, subset DP for Hamilton.
 parent=list(range(n));cycle=False
 def root(u):
  while parent[u]!=u:u=parent[u]
  return u
 for u,v,*w in edges:
  a,b=root(u),root(v)
  if a==b:cycle=True
  else:parent[a]=b
 if task=='connectivity':return root(source)==root(target)
 if task=='cycle':return cycle
 if task=='shortest_path':
  d=[math.inf]*n;d[source]=0
  for _ in range(n-1):
   for u,v,w in edges:d[v]=min(d[v],d[u]+w);d[u]=min(d[u],d[v]+w)
  return d[target]
 if task=='hamilton':
  adj=[0]*n
  for u,v in edges:adj[u]|=1<<v;adj[v]|=1<<u
  dp=[0]*(1<<n)
  for u in range(n):dp[1<<u]=1<<u
  for mask,ends in enumerate(dp):
   while ends:
    bit=ends & -ends;u=bit.bit_length()-1;ends-=bit
    options=adj[u]&~mask
    while options:
     v=options&-options;options-=v;dp[mask|v]|=v
  return bool(dp[-1])
 if task=='topology':
  indeg=[0]*n;adj=[[] for _ in range(n)]
  for u,v in edges:indeg[v]+=1;adj[u].append(v)
  queue=[u for u in range(n) if not indeg[u]];done=[]
  while queue:
   u=queue.pop();done.append(u)
   for v in adj[u]:
    indeg[v]-=1
    if not indeg[v]:queue.append(v)
  return len(done)==n

def independent_path(c,path):
 g=c['graph_data'];n=g['num_nodes'];edges=g['edges'];t=c['task']
 if not path or len(set(path))!=len(path) or any(v<0 or v>=n for v in path):return False
 if t=='topology':
  seen=set()
  for v in path:
   if any(b==v and a not in seen for a,b in edges):return False
   seen.add(v)
  return all(a in seen and b in seen for a,b in edges)
 e={frozenset((u,v)):w[0] if w else 1 for u,v,*w in edges}
 if any(frozenset((a,b)) not in e for a,b in zip(path,path[1:])):return False
 if t=='hamilton':return len(path)==n
 q=c['query_params'];s,d=q['source'],q['target']
 return {path[0],path[-1]}=={s,d} and sum(e[frozenset((a,b))] for a,b in zip(path,path[1:]))==independent_truth(t,n,tuple(map(tuple,edges)),s,d)

def main():
 inp=HERE/'scoring_audit_inputs.json.gz';data=json.loads(gzip.decompress(inp.read_bytes()));cases=data['cases']
 before=json.loads((HERE/'before-false-negative-review/audited_final_scores.json').read_text())
 reviews=json.loads((HERE/'manual_answer_review.json').read_text());lookup={(r['model'],r['case_id']):r for r in reviews}
 assert len(lookup)==len(reviews)
 excluded={str(c['graph_id']) for c in cases if c['task']=='hamilton' and set(v for e in c['graph_data']['edges'] for v in e)!=set(range(c['graph_data']['num_nodes']))}
 assert len(excluded)==62
 def include(c):return not(c['task']=='hamilton' and str(c['graph_id']) in excluded)
 unique={}
 graph_specs={}
 for c in cases:
  g=c['graph_data'];t=c['task'];q=c['query_params'];n=g['num_nodes'];e=tuple(map(tuple,g['edges']));s=q.get('source',0);d=q.get('target',0)
  key=(t,c['graph_id'],c['question_id'])
  spec=(n,tuple(sorted(tuple(sorted(edge[:2]))+tuple(edge[2:]) if t!='topology' else tuple(edge) for edge in g['edges'])),s,d)
  if key in graph_specs:assert graph_specs[key]==spec,(key,'graph differs across conditions')
  graph_specs[key]=spec
  if key in unique:continue
  truth=independent_truth(t,n,e,s,d);graph=graph_for(t,n,e)
  expected=reachable(graph,s,d) if t=='connectivity' else has_cycle(graph) if t=='cycle' else minimum_weight(n,e,s,d) if t=='shortest_path' else hamiltonian(n,e) if t=='hamilton' else True
  assert truth==expected,(key,truth,expected)
  if t=='shortest_path':assert math.isfinite(truth)
  unique[key]=truth
 out={k:v for k,v in before.items() if k!='models'}
 out.update(created_utc=datetime.now(timezone.utc).isoformat(),source=str(inp.relative_to(HERE.parent)),source_sha256=digest(inp),scorer_sha256=digest(HERE/'score_answers.py'),review_sha256=digest(HERE/'manual_answer_review.json'),review_script_sha256=digest(HERE/'review_scores.py'),excluded_hamilton_graph_ids=sorted(excluded,key=int),models={})
 audit={'excluded_hamilton_graph_ids':out['excluded_hamilton_graph_ids'],'independent_graph_checks':len(unique),'models':{},'changes':[],'unextracted':[],'ambiguous':[]}
 for m,md in data['models'].items():
  legacy_scores=dict(md['previous_scores'])
  old=dict(before['models'][m]['scores']);cells=defaultdict(Counter);counts=Counter();scores=[];archived=[]
  for c in cases:
   k=c['case_id'];r=md['responses'][k];checked,kind,ans=adjudicate(c,r['text']);review=lookup.get((m,k))
   if review:
    assert hashlib.sha256(r['text'].encode()).hexdigest()==review['response_sha256'],(m,k,'review text mismatch')
    assert review['status'] in ['correct_final_answer','reviewed_wrong_answer','incorrect_final_answer','ambiguous']
   old_check,_,_=adjudicate(c,r['text'],legacy=True)
   assert (r['release_correct'] if old_check is None else int(old_check))==legacy_scores[k],(m,k,'legacy mismatch')
   if review and review['status'] in ['correct_final_answer','reviewed_wrong_answer']:
    ans=review['answer'];kind='reviewed_explicit_answer'
    if isinstance(ans,list):checked=validate_path(c,ans)
    elif c['task']=='hamilton':checked=not hamiltonian(c['graph_data']['num_nodes'],tuple(map(tuple,c['graph_data']['edges'])))
    elif c['task']=='shortest_path':checked=not math.isfinite(unique[c['task'],c['graph_id'],c['question_id']])
    elif c['task']=='topology':checked=not unique[c['task'],c['graph_id'],c['question_id']]
    else:checked=ans==unique[c['task'],c['graph_id'],c['question_id']]
    assert bool(checked)==(review['status']=='correct_final_answer'),(m,k,ans)
   elif review and review['status']=='ambiguous':checked=False;kind='ambiguous_answer';ans=None
   elif review and review['status']=='incorrect_final_answer':assert checked is False,(m,k,'reviewed wrong final answer received credit')
   if isinstance(ans,(list,tuple)):
    assert checked==independent_path(c,ans),(m,k,ans)
   value=int(bool(checked));archived.append([k,value])
   if not include(c):continue
   scores.append([k,value]);counts.update(n=1,previous=old[k],score=value,gained=value>old[k],lost=value<old[k],unextracted=checked is None,ambiguous=kind=='ambiguous_answer',tool_only=kind=='tool_call_without_answer')
   cells[c['task'],c['order'],c['prompt_style']].update(n=1,score=value,previous=old[k])
   row=dict(model=m,case_id=k,previous=old[k],score=value,answer=ans,kind=kind,text=r['text'])
   if value!=old[k]:audit['changes'].append(row)
   if checked is None:audit['unextracted'].append(row)
   if kind=='ambiguous_answer':audit['ambiguous'].append(row)
  assert len(scores)==15950 and all(v['n']==(78 if t=='hamilton' else 140) for (t,o,p),v in cells.items())
  out['models'][m]={'scores':scores,'archived_scores':archived,'repaired':before['models'][m]['repaired'],'cells':[dict(task=t,order=o,prompt=p,**v) for (t,o,p),v in sorted(cells.items())]}
  audit['models'][m]=dict(counts);print(m,dict(counts))
 (HERE/'audited_final_scores.json').write_text(json.dumps(out,indent=2)+'\n')
 (HERE/'false_negative_review.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('Independent graph ground-truth checks',len(unique),'changes',len(audit['changes']),'unextracted',len(audit['unextracted']))
if __name__=='__main__':main()
