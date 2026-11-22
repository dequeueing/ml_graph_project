"""Screen every zero for valid candidate answers. Flags require semantic review.
This deliberately permissive scanner never changes any score.
"""
from pathlib import Path
import gzip,json,re,sys
from collections import Counter
from score_answers import adjudicate,answer_text,validate_path,hamiltonian
HERE=Path(__file__).resolve().parent
raw=json.loads(gzip.decompress((HERE/'scoring_audit_inputs.json.gz').read_bytes()))
scores=json.loads((HERE/'audited_final_scores.json').read_text())
cases={c['case_id']:c for c in raw['cases']}
# Includes mathematical spacers, annotations, dash/arrow variants and text lists.
seq=re.compile(r'\d+(?:\s*(?:->|,|[-–−])\s*\d+)+|\d+(?:[ \t]+\d+)+')
flags=[];counts=Counter();catalog=[]
for model,data in scores['models'].items():
 for key,score in data['scores']:
  if score:continue
  c=cases[key];r=raw['models'][model]['responses'][key];t=c['task'];text=r['text']
  checked,kind,selected=adjudicate(c,text)
  norm=answer_text(text)
  norm=re.sub(r'\s*\((?:weight\s*)?\d+\)\s*(?=->)', ' ',norm)
  norm=norm.replace('\\,',' ').replace('\\quad',' ').replace('\\;',' ').replace('⟶','->').replace('⇒','->')
  valid=[];visible_valid=[]
  for match in seq.finditer(norm):
   path=tuple(map(int,re.findall(r'\d+',match[0])))
   if t not in ['connectivity','cycle'] and validate_path(c,path):valid.append({'path':path,'context':norm[max(0,match.start()-100):match.end()+160]})
   if t=='hamilton':
    vertices={v for e in c['graph_data']['edges'] for v in e[:2]}
    if len(path)==len(set(path)) and set(path)==vertices:
     edges={tuple(e[:2]) for e in c['graph_data']['edges']}
     if all((u,v) in edges or (v,u) in edges for u,v in zip(path,path[1:])):
      visible_valid.append(path)
  istool='DSML' in text or '<tool_call' in text
  negative_possible=t=='hamilton' and not hamiltonian(c['graph_data']['num_nodes'],tuple(map(tuple,c['graph_data']['edges']))) and re.search(r'\bno\b|not exist|impossible|cannot|can.t',norm,re.I)
  reason=[]
  if valid and not istool:reason.append('graph_valid_candidate')
  if visible_valid and not valid and not istool:reason.append('visible_vertex_path_only')
  if negative_possible and kind!='explicit_negative_hamiltonian':reason.append('negative_answer_candidate')
  if t in ['connectivity','cycle']:reason.append('binary_zero')
  if checked is None and not istool:reason.append('unextracted_zero')
  counts[model,t,'total']+=1
  counts[model,t,kind]+=1
  row={'model':model,'case_id':key,'task':t,'selected':selected,'extraction':kind,'reasons':reason,'valid_candidates':valid,'visible_candidates':visible_valid,'text':text}
  catalog.append(row)
  if reason:flags.append(row)
result={'zero_count':len(catalog),'flag_count':len(flags),'counts':{'/'.join(k):v for k,v in counts.items()},'flags':flags}
(HERE/'false_negative_screen.json').write_text(json.dumps(result,indent=2))
(HERE/'zero_score_catalog.json').write_text(json.dumps(catalog,indent=2))
print('zeros',len(catalog),'flags',len(flags))
print(Counter(reason for row in flags for reason in row['reasons']))
print('by task',Counter(r['task'] for r in flags))
