"""Verify evidence and regenerate correlations using Python's standard library."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read_csv(name):
 with (ROOT/name).open() as f:return list(csv.DictReader(f))

def ranks(values):
 order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.0]*len(values);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and values[order[j]]==values[order[i]]:j+=1
  for k in range(i,j):result[order[k]]=(i+1+j)/2
  i=j
 return result

def spearman(x,y):
 x,y=ranks(x),ranks(y);xm,ym=statistics.mean(x),statistics.mean(y)
 return sum((a-xm)*(b-ym) for a,b in zip(x,y))/math.sqrt(sum((a-xm)**2 for a in x)*sum((b-ym)**2 for b in y))

def main():
 inventory=read_csv('architectures.csv');ids=[r['architecture_id'] for r in inventory]
 assert ids==[f'A{i:02d}' for i in range(1,21)]
 runs=read_csv('training_runs.csv');segments=0
 for item in inventory:
  assert (ROOT/item['configuration']).exists()
  endpoints=[]
  for run in [r for r in runs if r['architecture_id']==item['architecture_id']]:
   final=None
   for file in run['log_files'].split(';'):
    segments+=1
    for line in (ROOT/file).read_text().splitlines():
     r=json.loads(line)
     assert r['mode'] in ('train','val')
     if r['mode']=='val' and r.get('epoch')==100:final=float(r['accuracy_top-1'])
   assert final is not None
   assert math.isclose(final,float(run['epoch100_top1']),abs_tol=1e-10)
   endpoints.append(final)
  assert len(endpoints)==int(item['training_runs'])
  assert math.isclose(statistics.mean(endpoints),float(item['final_top1']),abs_tol=1e-9)
 scores=read_csv('proxy/scores.csv');definitions=json.loads((ROOT/'proxy/score_definitions.json').read_text())
 records=[json.loads(l) for l in (ROOT/'proxy/audit_records.jsonl').read_text().splitlines()]
 audits={(r['architecture_id'],r['model_seed']):r for r in records}
 assert len(audits)==len(records)==60
 expected={(i,s,d['proxy']) for i in ids for s in (11,17,23) for d in definitions}
 index={(r['architecture_id'],int(r['scoring_seed']),r['proxy']):float(r['score']) for r in scores}
 assert len(scores)==len(index)==540 and set(index)==expected
 for i,s,p in expected:
  a=audits[i,s];assert not a['errors'] and a['all_stages_completed']
  assert a['initial_state_sha256']==a['final_state_sha256']
  v=a
  for k in next(d['audit_field'] for d in definitions if d['proxy']==p):v=v[k]
  assert math.isfinite(index[i,s,p]) and index[i,s,p]==float(v)
 y=[float(r['final_top1']) for r in inventory];rows=[]
 for d in definitions:
  p=d['proxy'];rhos=[spearman([index[i,s,p] for i in ids],y) for s in (11,17,23)]
  rows.append(dict(proxy=p,n=20,rho_seed11=rhos[0],rho_seed17=rhos[1],rho_seed23=rhos[2],rho_mean=statistics.mean(rhos),rho_sample_sd=statistics.stdev(rhos)))
 with (ROOT/'correlations.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 table='| Proxy | Spearman ρ (mean ± SD) |\n|---|---:|\n'
 for r in rows:table+=f"| {r['proxy']} | {r['rho_mean']:.3f} ± {r['rho_sample_sd']:.3f} |\n"
 (ROOT/'table.md').write_text(table)
 # Hashes apply to anonymized release files, not to unmodified server originals.
 checks=json.loads((ROOT/'checksums.json').read_text()) if (ROOT/'checksums.json').exists() else {}
 for name,digest in checks.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 print(table)
 print(json.dumps(dict(architectures=len(ids),training_runs=len(runs),log_segments=segments,scoring_records=len(records),proxy_scores=len(scores),training_targets_verified=True,raw_proxy_fields_verified=True,checksums_verified=len(checks)),indent=2))

if __name__=='__main__':main()
