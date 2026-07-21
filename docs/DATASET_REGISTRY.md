# PART E — Dataset registry (sources, licenses, honest label reality)

Machine-readable source of truth: [`data/dataset_registry.json`](../data/dataset_registry.json). This is the
human summary. **Rule enforced here:** a dataset is only listed as *supporting* reply-prediction / causal
best-action / exposure modeling when the required labels **actually exist** — reconstructable ≠ labeled, and
observational click logs ≠ causal headline effects. `access_status` is a Jan-2026 knowledge snapshot and MUST
be re-verified before use.

## The P0 set — build the first benchmarks on these

| Capability | Dataset | Why P0 | The catch |
|---|---|---|---|
| **Best headline (causal)** | **Upworthy Research Archive** | ~32k **randomized** headline A/B tests, real clicks, CC-BY, open | clicks≠conversions; 2013-15 viral-era US |
| **Best action / uplift (causal)** | **Criteo Uplift** | ~13.9M rows, **randomized** treatment + conversion; the uplift-calibration bar | features anonymized (no text); non-commercial |
| **Diffusion / cascades + graph** | **Higgs Twitter (SNAP)** | retweet cascades **+ follower graph** + timing; lets us test graph-vs-no-graph | one event, no tweet text |

Upworthy doubles as the **Part I7 TRIBE representation ablation** benchmark (headline → click, with the plain
Llama-3.2-3B embedding as the control TRIBE must beat).

## Full registry by capability

**Individual response (Part F).** `Enron` (P1): public, ~500k msgs — reply-occurrence + reply-**delay** are
*reconstructable* from thread headers, with rich relationship history; but **no valence/objection/
meeting-booked labels** (would need annotation) and no cold-outreach. `Avocado` (P2): richer (calendar may
proxy meeting-booked) but **LDC-paid, non-commercial, no redistribution**. → *We can benchmark reply/latency
now; valence & meeting-booked need labeling or a new source; cold-outreach (the product's headline use) is
NOT in these corpora — a real gap to fill.*

**Message / headline selection (Part G).** `Upworthy` (P0, randomized/causal). `MIND` (P1): 15M+ click
impressions but **observational** — position/personalization confounds mean it supports content→click
*modeling*, **not** causal best-headline.

**Customer treatment / uplift (Part G).** `Criteo Uplift` (P0, randomized) and `Hillstrom` (P1, randomized,
tiny/clean) — both give the causal treatment-effect calibration the best-action arms need; neither has message
text (arm/feature only).

**Reddit / social-post outcomes.** `Pushshift` (P2): exact text + author history + score trajectory, but
**access is restricted since the 2023 Reddit API changes** and redistribution is legally fraught — a lawful
data path is the blocker, not the labels.

**Diffusion / virality (Part H).** `Higgs Twitter` (P0, graph+cascade+timing). `SEISMIC` (P1): the
**point-process (Hawkes) baseline** every agent-cascade arm must beat. `MemeTracker` (P1): content→spread
**without** a social graph (isolates content signal). `Weibo cascades` (P1): depth/breadth trees. `FakeNewsNet`
(P2): content+graph+veracity but **Twitter/X hydration is now paid/restricted** — hard blocker.

**Public-opinion change.** `ANES` panels (P2): opinion change with demographic breakdown, but **wave
granularity** — no per-event exposure timing. No clean public "event → pre/post sentiment by group with
exposure timing" set exists off the shelf; this capability is data-limited.

## Honest gaps this registry exposes

1. **Cold-outreach reply** (the product's individual-prediction use) has **no public labeled dataset** — email
   corpora are internal comms. This is the biggest data gap for Part F.
2. **Valence / objection / meeting-booked** reply labels don't exist publicly without annotation.
3. **Causal best-action with message TEXT** — Upworthy (headlines) is the only clean public one; email/ad
   creative A/B with text + randomization is largely proprietary.
4. **Public-opinion exposure timing** is unavailable at event granularity publicly.

→ Sequencing: start P0 (Upworthy headline, Criteo uplift, Higgs cascade) because they are open, randomized/
graph-complete, and each maps to a distinct product claim; treat the gaps above as explicit "needs data"
items, not as things the current stack can already validate.
