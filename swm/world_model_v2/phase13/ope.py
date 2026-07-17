"""Phase 13 off-policy evaluation (Parts 26–29) — grade target policies on LOGGED real decisions
without simulating anything: IPS/SNIPS, direct method, doubly-robust with cross-fitting, and
per-decision IS / weighted sequential DR for logged decision SEQUENCES.

The real-intervention benchmark scores Phase 13 against decisions somebody actually took (Part 30C
ledger rows, imported decision logs). The estimand — "what would the target policy have earned on the
logged population?" — is counterfactual and only identified under explicit assumptions. This module
makes those assumptions load-bearing instead of implicit:

  IPS / SNIPS      unbiased iff every logged decision carries its TRUE logging propensity and the
                   target's actions have overlap (the logger could have taken them). High variance
                   under thin overlap; optional weight clipping trades variance for known bias.
  direct method    no propensities needed; biased exactly where the fitted reward model is wrong —
                   most dangerously on actions the log never shows (extrapolation is FLAGGED, never
                   hidden).
  doubly robust    DM + IPS control variate: consistent if EITHER the propensities OR the reward
                   model is right, with K-fold cross-fitting split by CLUSTER hash so no decision is
                   scored by a model fitted on its own cluster (own-fit bias).
  per-decision IS  sequential logs: the ratio at step t multiplies only ratios up to t (lower
  / sequential DR  variance than trajectory IS); `sequential_dr` is the weighted-DR (WDR) variant —
                   self-normalized per-step weights plus a fitted per-step value model as control
                   variate (any value model keeps consistency; a good one buys variance).

Honesty (the load-bearing part): every estimator REFUSES (ValueError) when its assumptions are
structurally unmet — missing/invalid propensities for the weighting estimators, degenerate cross-fit
folds — and every weight-based result carries overlap diagnostics (ESS, clipped share, zero-propensity
matches). `ess_fraction < 0.05` sets diagnostics["weak_overlap"]=True: a numerically fine-looking value
must not be silently trusted. CIs are CLUSTER bootstrap (Parts 32/36 "clustered CIs": decisions inside
one decision environment are correlated, so the ENVIRONMENT is the resampling unit), deterministic
under a fixed seed. Estimator disagreement is itself a diagnostic (`estimator_disagreement`): when
IPS-family and model-family values diverge beyond their CIs, at least one assumption set is wrong.

Pure stdlib. `logistic_fit` / `linear_fit` are small deterministic pure-python regressors (batch
gradient descent; closed-form ridge via Gaussian elimination) validated on synthetic ground truth in
tests — validating the ESTIMATORS is legitimately synthetic; the benchmark data is real.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

WEAK_OVERLAP_ESS_FRACTION = 0.05          # Part 26: below this, the estimate is flagged, not trusted
DIAG_CLIP = 10.0                          # reference clip level for the share_clipped diagnostic


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(60.0, z)))
    ez = math.exp(max(-60.0, z))
    return ez / (1.0 + ez)


# ---------------------------------------------------------------- direct-method regressors
@dataclass
class FitResult:
    """A fitted regressor: `weights[0]` is the intercept, `weights[1:]` align with the feature vector.
    `kind` selects the link ("linear" | "logistic")."""
    kind: str
    weights: list = field(default_factory=list)
    n_train: int = 0

    def predict(self, x: list) -> float:
        if len(x) + 1 != len(self.weights):
            raise ValueError(f"feature dim {len(x)} does not match fit dim {len(self.weights) - 1}")
        z = self.weights[0] + sum(w * xi for w, xi in zip(self.weights[1:], x))
        return _sigmoid(z) if self.kind == "logistic" else z


def _check_xy(X: list, y: list, name: str) -> int:
    if not X or len(X) != len(y):
        raise ValueError(f"{name} needs non-empty X, y of equal length (got {len(X)}, {len(y)})")
    d = len(X[0])
    if any(len(row) != d for row in X):
        raise ValueError(f"{name} needs rectangular X (first row has {d} features)")
    if any(not math.isfinite(float(v)) for v in y):
        raise ValueError(f"{name} refuses non-finite targets")
    return d


def _solve(a: list, b: list) -> list:
    """Gaussian elimination with partial pivoting on a copy of [A|b]. Deterministic; raises on a
    (near-)singular system instead of returning garbage — the caller should raise l2."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular normal equations — increase l2 or drop collinear features")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    w = [0.0] * n
    for r in range(n - 1, -1, -1):
        w[r] = (m[r][n] - sum(m[r][c] * w[c] for c in range(r + 1, n))) / m[r][r]
    return w


def linear_fit(X: list, y: list, *, l2: float = 1e-3) -> FitResult:
    """Closed-form ridge regression (intercept unpenalized) via the normal equations + Gaussian
    elimination. Deterministic; suitable for the small feature counts the benchmark uses."""
    d = _check_xy(X, y, "linear_fit") + 1                       # +1: intercept-first design column
    xtx = [[0.0] * d for _ in range(d)]
    xty = [0.0] * d
    for row, yi in zip(X, y):
        z = [1.0] + [float(v) for v in row]
        for i in range(d):
            xty[i] += z[i] * float(yi)
            for j in range(d):
                xtx[i][j] += z[i] * z[j]
    for i in range(1, d):                                        # ridge on slopes only
        xtx[i][i] += float(l2)
    return FitResult(kind="linear", weights=_solve(xtx, xty), n_train=len(y))


def logistic_fit(X: list, y: list, *, l2: float = 1e-3, epochs: int = 300, lr: float = 0.1) -> FitResult:
    """Full-batch gradient-descent logistic ridge (intercept unpenalized), weights initialized at 0 —
    deterministic. Targets may be {0,1} or soft labels in [0,1]."""
    d = _check_xy(X, y, "logistic_fit")
    if any(v < 0.0 or v > 1.0 for v in map(float, y)):
        raise ValueError("logistic_fit targets must lie in [0, 1]")
    w = [0.0] * (d + 1)
    n = float(len(y))
    for _ in range(int(epochs)):
        grad = [0.0] * (d + 1)
        for row, yi in zip(X, y):
            err = _sigmoid(w[0] + sum(wj * float(xj) for wj, xj in zip(w[1:], row))) - float(yi)
            grad[0] += err
            for j, xj in enumerate(row):
                grad[j + 1] += err * float(xj)
        w[0] -= lr * grad[0] / n
        for j in range(1, d + 1):
            w[j] -= lr * (grad[j] / n + float(l2) * w[j] / n)
    return FitResult(kind="logistic", weights=w, n_train=len(y))


# ---------------------------------------------------------------- logged-data plumbing
def make_featurizer(contexts: list):
    """Default featurizer built ONCE from the logged contexts (deterministic): dict contexts map to
    the sorted union of numeric keys (missing → 0.0); list/tuple contexts pass through; scalars wrap.
    Building from the union keeps every row the same width even when keys differ per row."""
    keys = sorted({k for c in contexts if isinstance(c, dict)
                   for k, v in c.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})

    def featurize(context):
        if isinstance(context, dict):
            return [float(context.get(k, 0.0) or 0.0) for k in keys]
        if isinstance(context, (list, tuple)):
            return [float(v) for v in context]
        return [float(context)]
    return featurize


def _action_probs(policy, context) -> dict:
    """Normalize the two supported policy shapes to {action: prob}. A dict return is a distribution
    (actions are hashable, so a dict cannot itself be an action); anything else is deterministic."""
    out = policy(context)
    if isinstance(out, dict):
        if not out:
            raise ValueError("stochastic policy returned an empty distribution")
        if any(p < 0.0 for p in out.values()):
            raise ValueError("stochastic policy returned a negative probability")
        s = sum(out.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"stochastic policy probabilities sum to {s:.6f}, not 1")
        return dict(out)
    return {out: 1.0}


def _check_decisions(decisions: list, name: str, *, need_propensity: bool) -> None:
    if not decisions:
        raise ValueError(f"{name}: no logged decisions to evaluate")
    bad = []
    for i, d in enumerate(decisions):
        if "action" not in d or "reward" not in d:
            raise ValueError(f"{name}: decision {i} lacks required keys 'action'/'reward'")
        if not math.isfinite(float(d["reward"])):
            raise ValueError(f"{name}: decision {i} has a non-finite reward")
        p = d.get("propensity")
        if p is None or not (0.0 < float(p) <= 1.0):
            bad.append(i)
    if need_propensity and bad:
        raise ValueError(
            f"{name} requires a logged propensity in (0, 1] for every decision; "
            f"{len(bad)}/{len(decisions)} rows are missing/invalid (first bad row {bad[0]}). "
            f"Refusing to estimate — recover the logging propensities or use direct_method.")


def _cluster_of(d: dict, i: int) -> str:
    c = d.get("cluster")
    return str(c) if c not in (None, "") else f"__row{i}"        # i.i.d. fallback, flagged by caller


def _group_by_cluster(items: list, clusters: list) -> dict:
    out: dict = {}
    for c, it in zip(clusters, items):
        out.setdefault(c, []).append(it)
    return out


def _ess(weights: list) -> float:
    s, s2 = sum(weights), sum(w * w for w in weights)
    return (s * s / s2) if s2 > 0 else 0.0


# ---------------------------------------------------------------- results + clustered bootstrap
@dataclass
class PolicyEvalResult:
    """One estimator's verdict on one target policy. `ci` is a 95% CLUSTER-bootstrap interval;
    `ess` is the effective sample size of the importance weights (== n for pure direct method);
    `diagnostics` carries the assumption evidence (overlap, clipping, extrapolation flags)."""
    value: float
    se: float
    ci: tuple
    n: int
    ess: float
    estimator: str
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"estimator": self.estimator, "value": round(self.value, 6), "se": round(self.se, 6),
                "ci": [round(self.ci[0], 6), round(self.ci[1], 6)], "n": self.n,
                "ess": round(self.ess, 1), "diagnostics": dict(self.diagnostics)}


def cluster_bootstrap_ci(values_by_cluster: dict, stat_fn, n_boot: int = 400, seed: int = 0) -> dict:
    """Resample CLUSTERS (decision environments) with replacement and recompute `stat_fn` over the
    concatenated per-decision payloads — Parts 32/36: within-environment decisions are correlated, so
    the environment is the exchangeable unit. Deterministic under `seed`. With a single cluster the
    bootstrap is degenerate (se=0, zero-width CI) and says so instead of pretending precision."""
    clusters = sorted(values_by_cluster, key=str)
    if not clusters:
        raise ValueError("cluster_bootstrap_ci: no clusters supplied")
    rng = random.Random(int(seed))
    stats = []
    for _ in range(int(n_boot)):
        sample: list = []
        for _ in clusters:
            sample.extend(values_by_cluster[clusters[rng.randrange(len(clusters))]])
        stats.append(float(stat_fn(sample)))
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[min(len(stats) - 1, int(math.ceil(0.975 * (len(stats) - 1))))]
    mean = sum(stats) / len(stats)
    se = math.sqrt(sum((s - mean) ** 2 for s in stats) / max(1, len(stats) - 1))
    return {"ci": (lo, hi), "se": se, "n_clusters": len(clusters), "n_boot": int(n_boot),
            "degenerate": len(clusters) < 2}


def _finish(value: float, payloads: list, clusters: list, stat_fn, *, n: int, ess: float,
            estimator: str, diagnostics: dict, seed: int = 0, n_boot: int = 400) -> PolicyEvalResult:
    boot = cluster_bootstrap_ci(_group_by_cluster(payloads, clusters), stat_fn,
                                n_boot=n_boot, seed=seed)
    diagnostics["n_clusters"] = boot["n_clusters"]
    if boot["degenerate"]:
        diagnostics["single_cluster_ci_degenerate"] = True
    if any(c.startswith("__row") for c in clusters):
        diagnostics["cluster_fallback_iid"] = True             # rows without a cluster resample alone
    ess_fraction = ess / max(1, n)
    diagnostics["ess_fraction"] = round(ess_fraction, 6)
    if ess_fraction < WEAK_OVERLAP_ESS_FRACTION:
        diagnostics["weak_overlap"] = True
    return PolicyEvalResult(value=value, se=boot["se"], ci=boot["ci"], n=n, ess=ess,
                            estimator=estimator, diagnostics=diagnostics)


# ---------------------------------------------------------------- IPS / SNIPS
def ips(decisions: list, policy, *, clip: float = None, self_normalized: bool = False,
        n_boot: int = 400, seed: int = 0) -> PolicyEvalResult:
    """Inverse-propensity scoring: value = mean of  π(a_i|x_i)/μ(a_i|x_i) · r_i.  `clip` caps the
    weights (variance ↓, bias ↑, both reported); `self_normalized=True` gives SNIPS (Σwr/Σw — bounded,
    biased O(1/n)). Refuses without full logged propensities."""
    name = "snips" if self_normalized else "ips"
    _check_decisions(decisions, name, need_propensity=True)
    weights, payloads, clusters = [], [], []
    n_clipped = 0
    for i, d in enumerate(decisions):
        w = _action_probs(policy, d.get("context")).get(d["action"], 0.0) / float(d["propensity"])
        if clip is not None and w > clip:
            w, n_clipped = float(clip), n_clipped + 1
        weights.append(w)
        payloads.append((w * float(d["reward"]), w))
        clusters.append(_cluster_of(d, i))
    n = len(decisions)
    n_matched = sum(1 for w in weights if w > 0)
    if self_normalized:
        if sum(weights) <= 0:
            raise ValueError("snips is undefined with zero total weight — the target policy never "
                             "matches a logged action (no overlap); see overlap_diagnostics()")
        point = sum(wr for wr, _ in payloads) / sum(weights)

        def stat(sample, _fallback=point):
            sw = sum(w for _, w in sample)
            return (sum(wr for wr, _ in sample) / sw) if sw > 0 else _fallback   # 0/0 draw → point
    else:
        point = sum(wr for wr, _ in payloads) / n

        def stat(sample):
            return sum(wr for wr, _ in sample) / max(1, len(sample))
    diags = {"clip": clip, "n_clipped": n_clipped, "share_clipped": round(n_clipped / n, 6),
             "n_matched": n_matched, "weight_max": round(max(weights), 6) if weights else 0.0}
    return _finish(point, payloads, clusters, stat, n=n, ess=_ess(weights), estimator=name,
                   diagnostics=diags, seed=seed, n_boot=n_boot)


def snips(decisions: list, policy, *, clip: float = None, n_boot: int = 400,
          seed: int = 0) -> PolicyEvalResult:
    """Self-normalized IPS (see `ips`)."""
    return ips(decisions, policy, clip=clip, self_normalized=True, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------- direct method
def _fit_reward_models(rows: list, featurize, model_fn) -> dict:
    """Per-action reward models on (features, reward). Actions with fewer rows than features+2 get an
    honest constant (their mean); the pack records what was thin so extrapolation stays visible."""
    by_action: dict = {}
    for d in rows:
        by_action.setdefault(d["action"], []).append((featurize(d.get("context")), float(d["reward"])))
    all_r = [float(d["reward"]) for d in rows]
    global_mean = sum(all_r) / len(all_r) if all_r else 0.0
    models, thin = {}, []
    for a, xy in by_action.items():
        X = [x for x, _ in xy]
        yv = [r for _, r in xy]
        if len(xy) >= len(X[0]) + 2:
            models[a] = model_fn(X, yv)
        else:
            mu = sum(yv) / len(yv)
            models[a] = FitResult(kind="const", weights=[mu], n_train=len(yv))
            thin.append(a)
    return {"models": models, "global_mean": global_mean, "thin_actions": sorted(map(str, thin)),
            "seen_actions": set(by_action)}


def _q(pack: dict, action, feats: list) -> float:
    m = pack["models"].get(action)
    if m is None:
        return pack["global_mean"]                               # unseen action → global mean, flagged
    if m.kind == "const":
        return m.weights[0]
    return m.predict(feats)


def _dm_value(pack: dict, policy, context, feats: list, unseen: set) -> float:
    v = 0.0
    for a, p in _action_probs(policy, context).items():
        if p <= 0.0:
            continue
        if a not in pack["seen_actions"]:
            unseen.add(a)
        v += p * _q(pack, a, feats)
    return v


def direct_method(decisions: list, policy, *, model_fn=linear_fit, featurize=None,
                  n_boot: int = 400, seed: int = 0) -> PolicyEvalResult:
    """Direct method: fit a reward model PER ACTION on the logged data, then average the model's
    prediction for the policy's chosen action(s) over the logged contexts. Needs no propensities —
    its assumption is the model, so extrapolation onto actions the log never contains is flagged
    (`unseen_action_warning`) and scored at the honest fallback (global mean), never silently.
    `model_fn(X, y) -> FitResult` (e.g. `linear_fit`, `logistic_fit`, or `functools.partial` thereof);
    `featurize(context) -> [float]` defaults to `make_featurizer` over the logged contexts.
    The bootstrap CI resamples clusters with the model held FIXED — it reflects population variation,
    not model-fit uncertainty (recorded in diagnostics)."""
    _check_decisions(decisions, "direct_method", need_propensity=False)
    featurize = featurize or make_featurizer([d.get("context") for d in decisions])
    pack = _fit_reward_models(decisions, featurize, model_fn)
    payloads, clusters = [], []
    unseen: set = set()
    for i, d in enumerate(decisions):
        payloads.append(_dm_value(pack, policy, d.get("context"), featurize(d.get("context")), unseen))
        clusters.append(_cluster_of(d, i))
    n = len(decisions)
    point = sum(payloads) / n
    diags = {"model": getattr(model_fn, "__name__", str(model_fn)),
             "thin_actions": pack["thin_actions"], "ci_ignores_model_fit_uncertainty": True}
    if unseen:
        diags["unseen_action_warning"] = True
        diags["unseen_actions"] = sorted(map(str, unseen))
    return _finish(point, payloads, clusters, lambda s: sum(s) / max(1, len(s)), n=n, ess=float(n),
                   estimator="direct_method", diagnostics=diags, seed=seed, n_boot=n_boot)


# ---------------------------------------------------------------- doubly robust (cross-fitted)
def _fold_assignment(clusters: list, k: int, name: str) -> dict:
    """Cluster → fold by sha256 hash (stable across runs/processes, unlike hash()). If the hash split
    leaves any TRAINING side empty, fall back to round-robin over sorted clusters; with <2 clusters
    cross-fitting cannot avoid own-fit bias, so refuse."""
    uniq = sorted(set(clusters))
    if len(uniq) < 2:
        raise ValueError(f"{name}: cross-fitting needs >= 2 clusters (got {len(uniq)}) — "
                         f"own-fit bias cannot be avoided; add cluster labels or more environments")
    folds = {c: int(hashlib.sha256(c.encode()).hexdigest(), 16) % k for c in uniq}
    used = set(folds.values())
    if len(used) < min(k, len(uniq)):                            # a fold's training set would be empty
        folds = {c: i % k for i, c in enumerate(uniq)}
        folds["__assignment"] = "round_robin_fallback"
    return folds


def doubly_robust(decisions: list, policy, *, model_fn=linear_fit, featurize=None, clip: float = None,
                  k: int = 2, n_boot: int = 400, seed: int = 0) -> PolicyEvalResult:
    """Doubly robust: DM baseline + importance-weighted residual,
        term_i = Σ_a π(a|x_i) q̂^{-f(i)}(x_i,a) + w_i (r_i − q̂^{-f(i)}(x_i,a_i)),
    with K-fold CROSS-FITTING split by cluster hash — each decision is scored by models fitted on the
    OTHER folds only, so the control variate is independent of the row it corrects (no own-fit bias).
    Consistent if either the propensities or the reward model is right; refuses without propensities."""
    _check_decisions(decisions, "doubly_robust", need_propensity=True)
    featurize = featurize or make_featurizer([d.get("context") for d in decisions])
    clusters = [_cluster_of(d, i) for i, d in enumerate(decisions)]
    folds = _fold_assignment(clusters, int(k), "doubly_robust")
    packs = {f: _fit_reward_models([d for d, c in zip(decisions, clusters) if folds[c] != f],
                                   featurize, model_fn)
             for f in range(int(k))}
    payloads, weights = [], []
    unseen: set = set()
    n_clipped = 0
    for d, c in zip(decisions, clusters):
        pack = packs[folds[c]]
        feats = featurize(d.get("context"))
        w = _action_probs(policy, d.get("context")).get(d["action"], 0.0) / float(d["propensity"])
        if clip is not None and w > clip:
            w, n_clipped = float(clip), n_clipped + 1
        weights.append(w)
        dm = _dm_value(pack, policy, d.get("context"), feats, unseen)
        payloads.append(dm + w * (float(d["reward"]) - _q(pack, d["action"], feats)))
    n = len(decisions)
    diags = {"k_folds": int(k), "clip": clip, "n_clipped": n_clipped,
             "share_clipped": round(n_clipped / n, 6),
             "n_matched": sum(1 for w in weights if w > 0),
             "model": getattr(model_fn, "__name__", str(model_fn)),
             "thin_actions": sorted({a for p in packs.values() for a in p["thin_actions"]}),
             "ci_ignores_model_fit_uncertainty": True}
    if folds.get("__assignment"):
        diags["fold_assignment"] = folds["__assignment"]
    if unseen:
        diags["unseen_action_warning"] = True
        diags["unseen_actions"] = sorted(map(str, unseen))
    return _finish(sum(payloads) / n, payloads, clusters, lambda s: sum(s) / max(1, len(s)), n=n,
                   ess=_ess(weights), estimator="doubly_robust", diagnostics=diags,
                   seed=seed, n_boot=n_boot)


# ---------------------------------------------------------------- overlap / clipping diagnostics
def overlap_diagnostics(decisions: list, policy) -> dict:
    """Part 26 assumption evidence, computed WITHOUT raising (this is the tool you run to find out why
    an estimator refused): ESS of the would-be IPS weights, propensity range, the share of matched
    weights above the reference clip (10), and `n_zero_propensity_matches` — decisions where the
    target policy puts probability on the LOGGED action but that action's logged propensity is
    zero/None, i.e. rows that break the weighting estimators outright."""
    if not decisions:
        raise ValueError("overlap_diagnostics: no logged decisions")
    weights = []
    n_zero_prop_matches = n_missing = n_matched = 0
    props = []
    for d in decisions:
        p = d.get("propensity")
        pi_a = _action_probs(policy, d.get("context")).get(d.get("action"), 0.0)
        if p is None or not (0.0 < float(p) <= 1.0):
            n_missing += 1
            if pi_a > 0.0:
                n_zero_prop_matches += 1
            continue
        props.append(float(p))
        w = pi_a / float(p)
        weights.append(w)
        n_matched += w > 0
    n = len(decisions)
    ess = _ess(weights)
    out = {"n": n, "n_matched": n_matched, "ess": round(ess, 2),
           "ess_fraction": round(ess / n, 6),
           "propensity_min": min(props) if props else None,
           "propensity_max": max(props) if props else None,
           "share_clipped": round(sum(1 for w in weights if w > DIAG_CLIP) / n, 6),
           "n_zero_propensity_matches": n_zero_prop_matches,
           "n_missing_propensity": n_missing}
    if ess / n < WEAK_OVERLAP_ESS_FRACTION:
        out["weak_overlap"] = True
    return out


def clipping_sensitivity(decisions: list, policy, clips: tuple = (2, 5, 10, 20, None)) -> dict:
    """IPS value at each clip level (None = unclipped). A value that moves materially across clip
    levels is riding on a few extreme weights — report the whole curve, not one point."""
    return {c: ips(decisions, policy, clip=c).value for c in clips}


# ---------------------------------------------------------------- sequential estimators
def _check_sequences(sequences: list, name: str) -> None:
    if not sequences:
        raise ValueError(f"{name}: no logged sequences to evaluate")
    for i, s in enumerate(sequences):
        steps = s.get("steps")
        if not steps:
            raise ValueError(f"{name}: sequence {i} has no steps")
        for t, st in enumerate(steps):
            if "action" not in st or "reward" not in st:
                raise ValueError(f"{name}: sequence {i} step {t} lacks 'action'/'reward'")
            p = st.get("propensity")
            if p is None or not (0.0 < float(p) <= 1.0):
                raise ValueError(
                    f"{name} requires a logged propensity in (0, 1] at every step; sequence {i} "
                    f"step {t} is missing/invalid. Refusing — sequential IS has no propensity-free "
                    f"fallback (fit a model and use sequential_dr only after recovering propensities)")


def _cum_weights(steps: list, policy) -> list:
    """Cumulative importance products ρ_{0:t} = Π_{u<=t} π(a_u|x_u)/μ_u — the per-decision IS weights."""
    out, cw = [], 1.0
    for st in steps:
        cw *= _action_probs(policy, st.get("context")).get(st["action"], 0.0) / float(st["propensity"])
        out.append(cw)
    return out


def per_decision_is(sequences: list, policy, *, gamma: float = 1.0, clip: float = None,
                    n_boot: int = 400, seed: int = 0) -> PolicyEvalResult:
    """Per-decision importance sampling for logged SEQUENCES:
        V̂ = mean over sequences of  Σ_t γ^t ρ_{0:t} r_t,
    where ρ_{0:t} multiplies only the ratios UP TO t (each reward is reweighted by the decisions that
    could have influenced it — strictly lower variance than whole-trajectory IS). `clip` caps the
    cumulative weight. ESS is computed on the horizon (final-step) weights — trajectory-level overlap."""
    _check_sequences(sequences, "per_decision_is")
    payloads, clusters, final_w = [], [], []
    n_clipped = 0
    for i, s in enumerate(sequences):
        cws = _cum_weights(s["steps"], policy)
        g = 0.0
        for t, (st, cw) in enumerate(zip(s["steps"], cws)):
            if clip is not None and cw > clip:
                cw, n_clipped = float(clip), n_clipped + 1
            g += (gamma ** t) * cw * float(st["reward"])
        payloads.append(g)
        final_w.append(min(cws[-1], clip) if clip is not None else cws[-1])
        clusters.append(_cluster_of(s, i))
    n = len(sequences)
    diags = {"gamma": gamma, "clip": clip, "n_clipped": n_clipped,
             "mean_horizon": round(sum(len(s["steps"]) for s in sequences) / n, 3),
             "n_matched": sum(1 for w in final_w if w > 0)}
    return _finish(sum(payloads) / n, payloads, clusters, lambda s: sum(s) / max(1, len(s)), n=n,
                   ess=_ess(final_w), estimator="per_decision_is", diagnostics=diags,
                   seed=seed, n_boot=n_boot)


def _fit_step_models(train_seqs: list, featurize, model_fn, gamma: float) -> dict:
    """Per-(step, action) value models q̂_t(x, a) fitted on the logged RETURN-TO-GO Σ_{u>=t} γ^u r_u
    (the behavior policy's Q — any fixed q̂ keeps DR consistency; quality only buys variance).
    Thin cells fall back to the per-action pooled model, then to the global mean — recorded."""
    cells: dict = {}
    pooled: dict = {}
    all_g = []
    for s in train_seqs:
        rewards = [float(st["reward"]) for st in s["steps"]]
        g = 0.0
        gos = [0.0] * len(rewards)
        for t in range(len(rewards) - 1, -1, -1):
            g = rewards[t] + gamma * g
            gos[t] = g
        for t, st in enumerate(s["steps"]):
            row = (featurize(st.get("context")), gos[t])
            cells.setdefault((t, st["action"]), []).append(row)
            pooled.setdefault(st["action"], []).append(row)
            all_g.append(gos[t])
    global_mean = sum(all_g) / len(all_g) if all_g else 0.0

    def _fit(xy):
        X = [x for x, _ in xy]
        yv = [v for _, v in xy]
        if len(xy) >= len(X[0]) + 2:
            return model_fn(X, yv)
        return FitResult(kind="const", weights=[sum(yv) / len(yv)], n_train=len(yv))

    models = {key: _fit(xy) for key, xy in cells.items()}
    pooled_models = {a: _fit(xy) for a, xy in pooled.items()}
    thin = sorted(f"t{t}:{a}" for (t, a), m in models.items() if m.kind == "const")
    return {"models": models, "pooled": pooled_models, "global_mean": global_mean,
            "thin_cells": thin, "seen_actions": set(pooled)}


def _q_step(pack: dict, t: int, action, feats: list) -> float:
    m = pack["models"].get((t, action)) or pack["pooled"].get(action)
    if m is None:
        return pack["global_mean"]
    if m.kind == "const":
        return m.weights[0]
    return m.predict(feats)


def _wdr_stat(payloads: list) -> float:
    """Weighted (self-normalized) sequential DR over per-sequence payloads
    (cum-weights, rewards, q̂(x_t,a_t), v̂_π(x_t)):
        WDR = Σ_t Σ_i [ w_t^i r_t^i − w_t^i q̂(x_t^i,a_t^i) + w_{t−1}^i v̂_π(x_t^i) ],
    w_t^i = ρ^i_{0:t}/Σ_j ρ^j_{0:t}, w_{−1}^i = 1/n. Ragged horizons use the absorbing-state
    convention: finished trajectories keep ρ constant (they stay in the normalizer) and contribute
    zero reward/correction. A step where every trajectory has zero weight contributes nothing —
    that region of the target policy is simply unobserved (weak overlap shows in the ESS)."""
    n = len(payloads)
    if n == 0:
        return 0.0
    horizon = max(len(p["cum"]) for p in payloads)

    def cw(p, t):                                              # absorbing-state cumulative weight
        return p["cum"][t] if t < len(p["cum"]) else p["cum"][-1]

    total = 0.0
    z_prev = None
    for t in range(horizon):
        z = sum(cw(p, t) for p in payloads)
        for p in payloads:
            if t >= len(p["cum"]):
                continue                                       # finished: zero reward, zero correction
            w_prev = (1.0 / n) if t == 0 else (cw(p, t - 1) / z_prev if z_prev > 0 else 0.0)
            w_t = cw(p, t) / z if z > 0 else 0.0
            total += w_t * p["r"][t] - w_t * p["qa"][t] + w_prev * p["vpi"][t]
        z_prev = z
    return total


def sequential_dr(sequences: list, policy, *, model_fn=linear_fit, featurize=None,
                  gamma: float = 1.0, k: int = 2, n_boot: int = 400,
                  seed: int = 0) -> PolicyEvalResult:
    """Weighted doubly-robust (WDR) evaluation of logged sequences: per-decision importance weights,
    self-normalized per step across trajectories, with a cross-fitted per-step value model as control
    variate (see `_wdr_stat` for the exact form). The value model is fitted per (step, action) on
    return-to-go, split K ways by cluster hash so no sequence is corrected by a model fitted on its
    own cluster. Refuses without step propensities or with <2 clusters. The bootstrap recomputes the
    FULL self-normalized statistic per resample (normalization couples trajectories, so per-sequence
    means would be wrong); models stay fixed across resamples (recorded)."""
    _check_sequences(sequences, "sequential_dr")
    featurize = featurize or make_featurizer([st.get("context") for s in sequences
                                              for st in s["steps"]])
    clusters = [_cluster_of(s, i) for i, s in enumerate(sequences)]
    folds = _fold_assignment(clusters, int(k), "sequential_dr")
    packs = {f: _fit_step_models([s for s, c in zip(sequences, clusters) if folds[c] != f],
                                 featurize, model_fn, gamma)
             for f in range(int(k))}
    payloads, final_w = [], []
    unseen: set = set()
    for s, c in zip(sequences, clusters):
        pack = packs[folds[c]]
        cws = _cum_weights(s["steps"], policy)
        qa, vpi = [], []
        for t, st in enumerate(s["steps"]):
            feats = featurize(st.get("context"))
            qa.append(_q_step(pack, t, st["action"], feats))
            v = 0.0
            for a, p in _action_probs(policy, st.get("context")).items():
                if p <= 0.0:
                    continue
                if a not in pack["seen_actions"]:
                    unseen.add(a)
                v += p * _q_step(pack, t, a, feats)
            vpi.append(v)
        payloads.append({"cum": cws, "r": [float(st["reward"]) for st in s["steps"]],
                         "qa": qa, "vpi": vpi})
        final_w.append(cws[-1])
    n = len(sequences)
    diags = {"gamma": gamma, "k_folds": int(k),
             "model": getattr(model_fn, "__name__", str(model_fn)),
             "thin_cells": sorted({c for p in packs.values() for c in p["thin_cells"]}),
             "n_matched": sum(1 for w in final_w if w > 0),
             "ci_ignores_model_fit_uncertainty": True}
    if folds.get("__assignment"):
        diags["fold_assignment"] = folds["__assignment"]
    if unseen:
        diags["unseen_action_warning"] = True
        diags["unseen_actions"] = sorted(map(str, unseen))
    return _finish(_wdr_stat(payloads), payloads, clusters, _wdr_stat, n=n, ess=_ess(final_w),
                   estimator="sequential_dr", diagnostics=diags, seed=seed, n_boot=n_boot)


# ---------------------------------------------------------------- estimator agreement (Part 26)
def estimator_disagreement(results: list) -> dict:
    """Cross-estimator spread as evidence: max pairwise |Δvalue| (and which pair), plus whether every
    pair of 95% CIs overlaps. IPS-family and model-family estimators lean on DIFFERENT assumptions,
    so agreement is corroboration and CI-disjoint disagreement means at least one assumption fails —
    report it, don't average it away."""
    out = {"n_estimators": len(results),
           "values": {r.estimator: round(r.value, 6) for r in results}}
    if len(results) < 2:
        out.update({"max_abs_diff": 0.0, "max_pair": None, "all_cis_overlap": True,
                    "non_overlapping_pairs": []})
        return out
    max_diff, max_pair, disjoint = 0.0, None, []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            diff = abs(a.value - b.value)
            if diff >= max_diff:
                max_diff, max_pair = diff, (a.estimator, b.estimator)
            if a.ci[0] > b.ci[1] or b.ci[0] > a.ci[1]:
                disjoint.append((a.estimator, b.estimator))
    out.update({"max_abs_diff": round(max_diff, 6), "max_pair": max_pair,
                "all_cis_overlap": not disjoint, "non_overlapping_pairs": disjoint})
    return out
