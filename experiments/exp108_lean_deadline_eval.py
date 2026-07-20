"""LEAN eval: forecast = the §8-9 grounded DEADLINE-AWARE prior mean (build_outcome_rate_prior), no rich
rollout. Measures the lever's real lift on the SAME 25 questions, cheaply (~2 calls/q). Leakage-quarantined."""
import json, statistics
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from experiments.exp101_btf3_pilot import fetch_btf3, _forecast_input
from swm.api.deepseek_backend import default_chat_fn
from swm.world_model_v2.phase3_priors import build_outcome_rate_prior
from swm.world_model_v2.state import parse_time

ids = json.load(open('experiments/results/exp107_sample_ids.json'))
rows = {r['question_id']: r for r in fetch_btf3()}
llm = default_chat_fn(system="Reply ONLY JSON.", max_tokens=1500, temperature=0.2)

def work(qid):
    try:
        q = _forecast_input(rows[qid])
        as_of, hz = str(q['present_date'])[:10], str(q['expected_resolution_date'])[:10]
        plan = SimpleNamespace(question=q['question'], as_of=parse_time(as_of), horizon_ts=parse_time(hz),
                               provenance={'outcome_lean':'neutral','as_of':as_of})
        spec = build_outcome_rate_prior(plan, llm=llm)
        return {'qid':qid,'p':round(float(spec.mean),4),'outcome':int(rows[qid]['resolution']),
                'stage':spec.provenance.get('stage'),'src':spec.source_class,
                'sota':None if rows[qid].get('sota_forecast_probability') is None else round(float(rows[qid]['sota_forecast_probability'])/100,4)}
    except Exception as e:
        return {'qid':qid,'p':None,'outcome':int(rows[qid]['resolution']),'err':f"{type(e).__name__}"}

res=[]
with ThreadPoolExecutor(max_workers=4) as ex:
    for f in as_completed([ex.submit(work,q) for q in ids]): res.append(f.result())
sc=[r for r in res if r['p'] is not None]
ps=[r['p'] for r in sc]; ys=[r['outcome'] for r in sc]
def auc(ps,ys):
    P=[p for p,y in zip(ps,ys) if y==1]; N=[p for p,y in zip(ps,ys) if y==0]
    return round(sum((a>b)+0.5*(a==b) for a in P for b in N)/(len(P)*len(N)),4) if P and N else None
brier=round(sum((p-y)**2 for p,y in zip(ps,ys))/len(ps),4)
acc=round(sum((p>0.5)==y for p,y in zip(ps,ys))/len(ys),4)
ym=statistics.mean([p for p,y in zip(ps,ys) if y==1]); nm=statistics.mean([p for p,y in zip(ps,ys) if y==0])
sota=[(r['sota'],r['outcome']) for r in sc if r['sota'] is not None]
sb=round(sum((p-y)**2 for p,y in sota)/len(sota),4)
print(f"LEAN §8-9 deadline-prior forecast, {len(sc)}/25:")
print(f"  Brier {brier} | acc {acc} | AUC {auc(ps,ys)} | mean-p YES {ym:.3f} vs NO {nm:.3f}")
print(f"  vs EXP-107 rich 0.310/AUC0.413 | thin kernel 0.352/AUC0.521 | constant 0.240 | SOTA {sb}/{len(sota)}")
for r in sorted(sc,key=lambda r:r['p']): print(f"    {r['p']:.2f} y={r['outcome']} {r.get('stage')} | {rows[r['qid']]['question'][:48]}")
json.dump(res, open('experiments/results/lean_deadline_eval.json','w'), indent=1)
