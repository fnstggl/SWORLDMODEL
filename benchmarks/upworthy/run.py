"""Upworthy Research Archive harness — the interventional benchmark (audit KPI-A). IMPLEMENTED.

Randomized headline A/B tests, so the observed CTR difference between arms IS the causal effect of the
headline — choosing one is a real do(x). The harness lives in:
  - experiments/datasets_upworthy.py       (loader + committed parsed cache)
  - experiments/exp054_interventional.py    (policy-value / regret + CATE-sign KPI)

Run: python -m experiments.exp054_interventional
Data (14 MB, gitignored): curl -sSL -o data/upworthy_exploratory.csv https://osf.io/download/3vqmp/
Contamination caveat (public since 2021): treat as a MECHANISM benchmark (does the model pick the
causally-better headline), not a leakage-free skill number.
"""
from __future__ import annotations

IMPLEMENTED = True


def run():
    from experiments.exp054_interventional import run as _run
    return _run()


if __name__ == "__main__":
    run()
