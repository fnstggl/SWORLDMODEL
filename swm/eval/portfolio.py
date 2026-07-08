"""Portfolio backtest — run the no-cheat event backtest across many domains and MAP where fidelity wins.

The general social world model's honest scoreboard is not one number but a MATRIX: domain × fidelity ×
skill-vs-baselines. Some domains (modelable evolving populations, weak simple baselines) reward high-fidelity
simulation; others (strong momentum/market baselines) do not. This runs a set of `Domain` adapters — each
builds Questions + a forecaster at ≥1 fidelity level — through `event_backtest`, and summarizes:
  - does the high-fidelity model BEAT ALL baselines?
  - does adding fidelity RAISE the skill (the thesis, per domain)?
so we learn WHERE to reach for the rich simulation, from data, not assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from swm.eval.event_backtest import backtest


@dataclass
class Domain:
    """`build(fidelity) -> (questions, forecast_fn)`; `fidelities` ordered low→high (e.g. ('few','full') or
    (1, 5, 12))."""
    name: str
    build: object
    fidelities: tuple = ("few", "full")
    kind: str = ""                       # optional tag: 'population' | 'market' | 'institution' | ...


def _skill(card):
    sv = card.get("skill_vs", {})
    return sv.get("persistence", sv.get("base_rate"))


def run_portfolio(domains, *, check_asof=True) -> dict:
    results = {}
    for d in domains:
        per_fid = {}
        for fid in d.fidelities:
            try:
                qs, fc = d.build(fid)
            except Exception as e:                       # a domain that can't build must not kill the sweep
                per_fid[str(fid)] = {"error": str(e)[:120]}
                continue
            if qs:
                per_fid[str(fid)] = backtest(qs, fc, check_asof=check_asof)
        results[d.name] = {"kind": d.kind, "by_fidelity": per_fid}
    return {"domains": results, "map": _map(results)}


def _map(results) -> dict:
    out = {}
    for name, r in results.items():
        fids = [(k, v) for k, v in r["by_fidelity"].items() if "skill_vs" in v]
        if not fids:
            out[name] = {"status": "no_scored_fidelity"}
            continue
        lo_c, hi_c = fids[0][1], fids[-1][1]
        lo, hi = _skill(lo_c), _skill(hi_c)
        out[name] = {"kind": r["kind"], "n": hi_c.get("n"),
                     "low_fidelity_skill": lo, "high_fidelity_skill": hi,
                     "fidelity_helps": (hi is not None and lo is not None and hi > lo + 1e-6),
                     "beats_all_baselines": hi_c.get("beats_all_baselines"),
                     "skill_vs": hi_c.get("skill_vs")}
    return out


def summarize(portfolio) -> str:
    """One-line-per-domain readout of the fidelity map."""
    lines = ["domain                 kind         n     lo-fid   hi-fid   fidelity?  beats-all?"]
    for name, m in portfolio["map"].items():
        if m.get("status"):
            lines.append(f"{name:22s} {m['status']}")
            continue
        lines.append(f"{name:22s} {str(m.get('kind','')):12s} {str(m.get('n','')):5s} "
                     f"{_f(m['low_fidelity_skill'])}  {_f(m['high_fidelity_skill'])}  "
                     f"{str(m['fidelity_helps']):9s}  {m['beats_all_baselines']}")
    return "\n".join(lines)


def _f(x):
    return f"{x:+.3f}" if isinstance(x, (int, float)) else "  n/a "
