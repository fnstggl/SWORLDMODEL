# EXP-031 — Posterior-guided attribution training (the paper's recipe): an honest negative for our stack

The SWM paper's central training trick is **posterior-guided event attribution**: since nobody labels
"which event caused this belief shift," use a *hindsight* attributor (which sees the outcome) to
pseudo-label attribution, then train a *forward* attributor to match. We implemented it faithfully and
tested whether it improves our transition operator. **It does not — and the reason is instructive.**

## What we built
- `swm/transition/attribution.py`: a `ForwardAttributor` (P_η) — a per-news causal scorer trained by
  logistic regression against the hindsight attribution labels shipped in SWM-Bench (foresight learns
  from hindsight; features are strictly pre-shift, only the labels are hindsight, which is the point).
- Its transition-level "event strength" as a learned gate for the belief-transition operator, plus a
  purely-learned world model (no LLM at inference) with and without attribution-weighted news features.

## Result (Kalshi, no-cheat)
**Q1 — Does hindsight supervise a useful forward attributor?** Barely. On held-out transitions the
attributor's **AUC for ranking causal vs non-causal news is 0.586** (chance 0.5); mean score for causal
news 0.248 vs 0.222 for non-causal. Raw accuracy 0.929 only *looks* good because causal news are 6.7% of
the set (predicting "not causal" scores 0.933).

**Q2 — Does it improve the transition?** No.
| tier | MAE ↓ | 3-way DA ↑ |
|---|---|---|
| persistence | 0.0603 | 0.379 |
| **LLM event-impact channel (EXP-030)** | **0.0594** | **0.426** |
| posterior-gated LLM (learned gate × LLM direction) | 0.0597 | 0.393 |
| naive learned (no attribution, no LLM) | 0.0703 | 0.380 |
| posterior-guided learned (attribution, no LLM) | 0.0712 | 0.387 |

Gating the LLM channel by the learned attributor **hurts** (DA 0.393 < 0.426). The purely-learned models
lose to persistence. Posterior-guided learned edges naive learned on DA (0.387 vs 0.380) — the mechanism
is *directionally* present — but not enough to matter.

## Why it's a negative for us (the instructive part)
1. **Cheap forward features can't replicate large-LLM attribution.** The hindsight labels come from a
   frontier LLM's deep reading of each news item; our salience + result-word features rank causal news
   only slightly above chance (AUC 0.586). The paper's forward attributor is itself an LLM — with cheap
   features the recipe starves.
2. **Our one-shot LLM impact already attributes implicitly.** The EXP-030 event channel had an agent
   read *all* the candidate news and return a net signed impact — it already did the attribution inside
   that reasoning. A separate forward attributor is therefore redundant, and a noisy learned gate on top
   only degrades a good signal.

So posterior-guided attribution is the right idea *when your forward model is an LLM and you have no
per-transition event signal*. We have a strong per-transition LLM channel, which subsumes it.

## Decision (per the "don't merge a regression" rule)
The posterior-gated transition is **not** wired into the operator — it regresses directional accuracy.
The `ForwardAttributor` module is kept as an additive, reusable component (it does not change the winning
EXP-030 operator), and this negative is documented rather than hidden. The natural way to make the recipe
pay off — an **LLM-based per-news forward attributor** — is redundant with our one-shot channel, so we
did not pursue it.

## Reproduce
`python -m experiments.exp031_posterior_guided` (uses the committed EXP-030 impact signals + SWM-Bench).
`python -m pytest tests/test_attribution.py`.
