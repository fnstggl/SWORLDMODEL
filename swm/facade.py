"""The canonical runtime facade — the ONE public door for all forecasts and simulations.

Every run declares an explicit architecture identity; nothing silently selects a legacy engine:

    forecast(question, architecture="world_model_v2", ...)          # the development default, NOT yet product
    forecast(question, architecture="baseline:grounded_direct", ...) # explicit, product_eligible=False
    forecast(question, architecture="baseline:observer_panel_v1", ...)

The PRODUCT-ELIGIBILITY CONTRACT: every result carries {architecture, version, commit, plan_hash,
legacy_executed, baseline, product_eligible, validation_status}. A `world_model_v2` run that touched any
legacy engine FAILS (raises) rather than shipping a contaminated result. Baselines always carry
product_eligible=False. Until V2 passes its first real held-out reference-world benchmark, the stable product
default remains the v1 path — but reaching it requires NAMING it (`baseline:*`); ambiguity is an error.

FREEZE POLICY (see docs/LEGACY_ARCHITECTURE_MAP.md): V1 receives baseline-preservation and critical fixes
only. All new state/mechanisms/compilers belong in swm/world_model_v2/. An unsupported V2 capability abstains
or marks itself experimental — it never silently falls back to legacy machinery.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

ARCHITECTURES = ("world_model_v2", "baseline:grounded_direct", "baseline:direct_ensemble",
                 "baseline:observer_panel_v1", "baseline:society_v1", "baseline:parametric_v1")

V2_VALIDATION_STATUS = "architecture_validated"     # flips to benchmark labels as evidence lands


class ArchitectureError(RuntimeError):
    pass


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class RunRecord:
    architecture: str
    version: str = "v2.0-foundation"
    commit: str = field(default_factory=_commit)
    plan_hash: str = ""
    legacy_executed: bool = False
    baseline: str = ""
    product_eligible: bool = False
    validation_status: str = ""

    def finalize(self):
        if self.architecture == "world_model_v2":
            if self.legacy_executed:
                raise ArchitectureError(
                    "PRODUCT-ELIGIBILITY VIOLATION: a world_model_v2 run executed legacy code — the result "
                    "is contaminated and must not ship. Fix the leak; do not mark this eligible.")
            self.validation_status = self.validation_status or V2_VALIDATION_STATUS
            # V2 is NOT product-eligible until a reference world beats the fair grounded baseline held-out
            self.product_eligible = self.validation_status in ("statistically_supported",
                                                               "production_validated")
        else:
            self.product_eligible = False               # baselines are science, not product
            self.baseline = self.architecture
        return self

    def as_dict(self):
        return {"architecture": self.architecture, "version": self.version, "commit": self.commit,
                "plan_hash": self.plan_hash, "legacy_executed": self.legacy_executed,
                "baseline": self.baseline, "product_eligible": self.product_eligible,
                "validation_status": self.validation_status}


def forecast(question: str, *, architecture: str, llm=None, evidence: str = "", as_of: str = "",
             horizon: str = "", **kw) -> dict:
    """The one entry point. Downstream callers never import engines directly."""
    if architecture not in ARCHITECTURES:
        raise ArchitectureError(f"unknown architecture {architecture!r} — no ambiguous default exists; "
                                f"choose one of {ARCHITECTURES}")
    rec = RunRecord(architecture=architecture)
    if architecture == "world_model_v2":
        from swm.world_model_v2.compiler import CompileAbstention, compile_world
        from swm.world_model_v2.materialize import run_from_plan
        try:
            plan = compile_world(question, llm=llm, evidence=evidence, as_of=as_of, horizon=horizon, **kw)
        except CompileAbstention as e:
            return {"question": question, "abstain": True, "abstain_reason": str(e),
                    "run": rec.finalize().as_dict()}
        rec.plan_hash = plan.provenance.get("prompt_hash", "")
        result, _ = run_from_plan(plan, llm=llm)
        return {"question": question, **result, "run": rec.finalize().as_dict()}
    # ---- explicit baselines (deprecated for new development; preserved for science) ----
    # EVERY name executes its OWN mechanism — call-spy tests pin label↔implementation identity (Part 0).
    rec.legacy_executed = True
    if architecture == "baseline:grounded_direct":
        # exactly ONE grounded direct forecast: ground once, one LLM call, no roles, no pooling
        from swm.engine.grounding import SceneGrounder
        from swm.eval.ablation import _p_single
        dossier = SceneGrounder(llm, today=as_of).ground(question, evidence=evidence or None)
        p = None if dossier.abstain else _p_single(llm, question, as_of, dossier)
        return {"question": question, "p": p, "abstain": p is None, "run": rec.finalize().as_dict()}
    if architecture == "baseline:direct_ensemble":
        # N INDEPENDENT grounded direct forecasts, log-linear pooled — no lenses, no personas, no anchor
        from swm.engine.grounding import SceneGrounder
        from swm.eval.tiered_ablation import _grounded_ensemble
        n = int(kw.pop("n_ensemble", 10))
        dossier = SceneGrounder(llm, today=as_of).ground(question, evidence=evidence or None)
        p = None if dossier.abstain else _grounded_ensemble(llm, question, as_of, dossier, n)
        return {"question": question, "p": p, "abstain": p is None, "n_ensemble": n,
                "pooling": "log_linear", "run": rec.finalize().as_dict()}
    if architecture == "baseline:parametric_v1":
        # the parametric/compiler kernel path — NEVER the observer panel
        from swm.engine.front_door import parametric_binary_p
        p = parametric_binary_p(question, as_of, llm)
        return {"question": question, "p": p, "abstain": p is None,
                "mechanism": "parametric_kernel", "run": rec.finalize().as_dict()}
    if architecture in ("baseline:observer_panel_v1", "baseline:society_v1"):
        from swm.engine.front_door import agent_world_model
        wm = agent_world_model()
        wm.event_engine = "society" if architecture == "baseline:society_v1" else "panel"
        wm.route_contests = False                          # the NAMED mechanism runs — no silent rerouting
        res = wm.simulate(question, as_of=as_of, binary=True, **kw)
        res["run"] = rec.finalize().as_dict()
        return res
    raise ArchitectureError(f"unhandled architecture {architecture!r}")
