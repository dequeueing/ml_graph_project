"""Audit every saved node-classification answer against the target label."""
import gzip,json,re,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from score_answers import normalized
HERE=Path(__file__).resolve().parent

def extract_node_label(text,target):
 text=normalized(text)
 if 'DSML' in text or '<tool_call' in text:return None,'tool_call'
 candidates=[]
 # Bare leading integer is a direct answer, not a neighboring vertex ID.
 for pattern in [r'^\s*([0-6])(?=\s|[.,]|$)',r'(?m)^\s*([0-6])[.]?\s*$',r'^\s*(?:label|class)\s*[:=]?\s*([0-6])\b',r'\b(?:final\s+)?answer\s*(?:is)?\s*[:=]\s*(?:label\s*)?([0-6])\b']:
  for m in re.finditer(pattern,text,re.I):candidates.append((m.start(),int(m[1]),'direct_label'))
 prefix=rf'(?:label\s+of\s+node\s+{target}\b|node\s+{target}(?:[’\x27]s)?(?:\s+label)?)'
 for m in re.finditer(prefix+r'\s*(?:(?:=|appears to have|is likely labeled|is also labeled|is most likely labeled|is predicted to have|is classified as|is|has|has the|would be|should be|can be predicted as|is predicted to be|most likely has|likely has|is likely|is most likely|is labeled|is labelled|is probably|:)\s*)+(?:the\s+)?(?:label|class)?\s*[:=]?\s*([0-6])\b',text,re.I):candidates.append((m.start(),int(m[1]),'target_label'))
 for m in re.finditer(r'\b(?:suggests?|predicted|inferred|likely|probable|best inference|best prediction|prediction|predict)(?:\s+(?:label|class))?\s*(?:is|:|would be)?\s*(?:label\s*)?([0-6])\b',text,re.I):candidates.append((m.start(),int(m[1]),'prediction_label'))
 for m in re.finditer(r'\b(?:its|the predicted|the inferred|the most likely|the majority|the most common neighboring|the) label\s+is\s*[:=]?\s*([0-6])\b',text,re.I):candidates.append((m.start(),int(m[1]),'concluding_label'))
 if candidates:
  _,label,kind=max(candidates)
  return label,kind
 if re.search(r'unknown|cannot|can.t|not (?:be )?determin|not enough|insufficient|enough information|unlabeled|doesn.t determine|not specified|no specified label|no known label|don.t determine|no rule|no determinable label|no stated label|indeterminate|undetermined|no inferable label|not assigned|not provided|\?',text,re.I):return None,'abstention'
 return None,'unextracted'

def main():
 data=json.loads(gzip.decompress((HERE/'verification_prompt_inputs.json.gz').read_bytes()))
 lookup={c['case_id']:c for c in data['node_cases']}
 output={'source':'analysis/verification_prompt_inputs.json.gz','source_sha256':hashlib.sha256((HERE/'verification_prompt_inputs.json.gz').read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'models':{},'changes':[],'unextracted':[]}
 for model,rows in data['node_models'].items():
  counts=Counter();scores=[];cells=defaultdict(Counter)
  for key,r in rows.items():
   c=lookup[key];g=c['graph_data'];target=g['test_node'];truth=g['node_labels'][str(target)]
   label,kind=extract_node_label(r['text'],target)
   value=int(label==truth) if label is not None else 0
   counts.update(n=1,previous=r['correct'],correct=value,unextracted=kind=='unextracted',abstention=kind=='abstention',tool_call=kind=='tool_call')
   cells[c['group'],c['order']].update(n=1,correct=value)
   scores.append([key,value])
   record={'model':model,'case_id':key,'target':target,'expected':truth,'previous':r['correct'],'score':value,'prediction':label,'kind':kind,'text':r['text']}
   if value!=r['correct']:output['changes'].append(record)
   if kind=='unextracted':output['unextracted'].append(record)
  output['models'][model]={'counts':dict(counts),'scores':scores,'cells':[{'group':g,'order':o,**v} for (g,o),v in cells.items()]}
  print(model,dict(counts))
 (HERE/'node_score_audit.json').write_text(json.dumps(output,indent=2))
 print('Changes',len(output['changes']),'unextracted',len(output['unextracted']))

if __name__=='__main__':main()
