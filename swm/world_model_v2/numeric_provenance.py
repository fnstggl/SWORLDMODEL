"""Runtime numeric-causal provenance gate — the NO-ARBITRARY-NUMERIC-REALITY contract (§NAP).

THE INVARIANT. A numerical value may affect a production world trajectory, actor decision, hazard,
terminal readout or recommendation ONLY if its causal provenance is one of the APPROVED_SOURCE_CLASSES
below. Everything else must be represented as a qualitative typed state, a structural hypothesis, a
concrete branch condition, a scenario-specific unresolved mechanism, an explicit ablation that cannot
contribute to the production answer, or an explicit `unresolved` / `under_modeled` record.

There is NO "assumed", "documented prior", "generic broad prior", "LLM-estimated", "neutral default"
or "reference-class guess" class. Sampling an invented prior does not make it real; widening an
invented coefficient does not make it empirical; labeling it honestly does not stop it from changing
the answer. Absence of a justified number means the mechanism is UNRESOLVED — never that its effect
is zero, one, or 0.5.

HOW IT WORKS.
  * Every operator that consumes a numeric value to alter world evolution registers it through
    `approve_numeric` / `reject_numeric` on the world's ledger (`ledger_of(world)`).
  * `approve_numeric` validates the declared source class and returns the value; a non-approved
    class raises `NumericProvenanceRejected` — the caller must then record the mechanism unresolved
    (`record_unresolved_mechanism`) and preserve the branch, never substitute a broad default.
  * Plan-level (conversion-time) decisions use a plan ledger (`plan_ledger_of(plan)`) so the compile
    stage's accept/reject decisions are auditable before any branch runs.
  * Every result carries a `numeric_causal_inputs` manifest (`merge_manifests` over branch ledgers)
    divided into approved-and-consumed / approved-unused / rejected / compute-safety-only /
    analysis-display-only / ablation-only.

FITTED-ARTIFACT ELIGIBILITY. `empirically_fitted_and_validated` is a claim that must be PROVEN by a
versioned parameter artifact: `fitted_artifact_eligible(pack)` checks training population, outcome
definition, frozen data cutoff, sample size, fitting procedure, held-out metrics, calibration
diagnostics, domain/transport restrictions and architecture compatibility. A JSON file existing is
not eligibility — the current family packs (keyword-selected reference classes, no held-out
evaluation, no transport check) FAIL this gate and are diagnostic-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: the ONLY source classes whose numbers may causally affect production execution
APPROVED_SOURCE_CLASSES = (
    "observed_measurement",                # a measured/reported real-world quantity with a source
    "explicit_user_input",                 # the caller stated the number
    "institutional_rule",                  # a literal legal/procedural threshold (votes needed, quorum)
    "physical_identity_or_conservation",   # arithmetic identity, conservation law, calendar arithmetic
    "empirically_fitted_and_validated",    # a fitted artifact that PASSES fitted_artifact_eligible
    "derived_deterministic",               # deterministic arithmetic over approved inputs only
)

#: classes that may exist in production but must NEVER alter the behavioral model — compute limits
#: must produce visible truncation; tolerances/display thresholds never touch world evolution
NON_CAUSAL_CLASSES = (
    "compute_safety",        # budgets, caps, retry limits — visible truncation only
    "numerical_tolerance",   # solver epsilons, convergence tolerances
    "analysis_display",      # rounding, report grids, UI thresholds
    "ablation_only",         # explicitly-named scientific ablation, never the production answer
)

#: rejected-source labels seen in the wild — recorded verbatim on rejection for the audit trail
KNOWN_ARBITRARY_CLASSES = (
    "documented_prior_unfitted", "llm_estimated", "neutral_default", "reference_class_guess",
    "broad_prior", "lean_beta", "keyword_family_rate", "sampled_unfitted_prior",
)


class NumericProvenanceRejected(Exception):
    """A load-bearing numeric input lacked approved provenance. The consuming mechanism must be
    recorded unresolved/under-modeled — never executed with a substituted default."""

    def __init__(self, message: str, *, name: str = "", consumer: str = ""):
        super().__init__(message)
        self.name, self.consumer = name, consumer


@dataclass
class CausalNumericInput:
    """One registered numeric real-world causal input (or a registered refusal)."""
    name: str
    value: object                       # float | int | list | tuple (distributions register bounds)
    units: str
    causal_role: str                    # the real concept this number claims to represent
    source_class: str
    consumer: str = ""                  # operator/function that consumes it
    evidence_id: str = ""               # evidence claim / parameter-artifact id
    fitted_on: dict = field(default_factory=dict)
    architecture_compat: str = ""
    applicability: str = ""             # applicability/transport verdict for fitted artifacts
    can_alter_terminal: bool = True
    production_eligible: bool = False
    rejection_reason: str = ""
    consumed: bool = False

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "units": self.units,
                "causal_role": self.causal_role, "source_class": self.source_class,
                "consumer": self.consumer, "evidence_id": self.evidence_id,
                "fitted_on": dict(self.fitted_on or {}),
                "architecture_compat": self.architecture_compat,
                "applicability": self.applicability,
                "can_alter_terminal": self.can_alter_terminal,
                "production_eligible": self.production_eligible,
                "rejection_reason": self.rejection_reason, "consumed": self.consumed}


class NumericProvenanceLedger:
    """The per-world (or per-plan) registry of every numeric causal input touched during execution."""

    def __init__(self):
        self.inputs: list = []

    # -------------------------------------------------------------- registration
    def approve(self, *, name: str, value, units: str, causal_role: str, source_class: str,
                consumer: str = "", evidence_id: str = "", fitted_on: dict = None,
                architecture_compat: str = "", applicability: str = "",
                can_alter_terminal: bool = True, consumed: bool = True):
        """Register an APPROVED numeric causal input and return its value. Raises
        NumericProvenanceRejected when the source class is not approved — the caller must then
        mark the mechanism unresolved instead of executing it."""
        if source_class in NON_CAUSAL_CLASSES:
            ent = CausalNumericInput(name=name, value=value, units=units, causal_role=causal_role,
                                     source_class=source_class, consumer=consumer,
                                     evidence_id=evidence_id, fitted_on=fitted_on or {},
                                     can_alter_terminal=False, production_eligible=True,
                                     consumed=consumed)
            self.inputs.append(ent)
            return value
        if source_class not in APPROVED_SOURCE_CLASSES:
            ent = CausalNumericInput(name=name, value=value, units=units, causal_role=causal_role,
                                     source_class=source_class, consumer=consumer,
                                     evidence_id=evidence_id, fitted_on=fitted_on or {},
                                     can_alter_terminal=can_alter_terminal,
                                     production_eligible=False,
                                     rejection_reason=f"source_class {source_class!r} is not an "
                                                      f"approved causal provenance", consumed=False)
            self.inputs.append(ent)
            raise NumericProvenanceRejected(
                f"numeric causal input {name!r} (consumer {consumer!r}) has source class "
                f"{source_class!r}, which may not alter production execution", name=name,
                consumer=consumer)
        if source_class == "empirically_fitted_and_validated":
            ok, why = fitted_artifact_eligible(fitted_on or {})
            if not ok:
                ent = CausalNumericInput(name=name, value=value, units=units,
                                         causal_role=causal_role, source_class=source_class,
                                         consumer=consumer, evidence_id=evidence_id,
                                         fitted_on=fitted_on or {},
                                         can_alter_terminal=can_alter_terminal,
                                         production_eligible=False,
                                         rejection_reason=f"fitted artifact ineligible: {why}",
                                         consumed=False)
                self.inputs.append(ent)
                raise NumericProvenanceRejected(
                    f"numeric causal input {name!r} claims a fitted artifact but the artifact is "
                    f"not production-eligible: {why}", name=name, consumer=consumer)
        ent = CausalNumericInput(name=name, value=value, units=units, causal_role=causal_role,
                                 source_class=source_class, consumer=consumer,
                                 evidence_id=evidence_id, fitted_on=fitted_on or {},
                                 architecture_compat=architecture_compat,
                                 applicability=applicability,
                                 can_alter_terminal=can_alter_terminal, production_eligible=True,
                                 consumed=consumed)
        self.inputs.append(ent)
        return value

    def reject(self, *, name: str, value, units: str, causal_role: str, source_class: str,
               consumer: str = "", why: str = "") -> None:
        """Record a numeric input that was REFUSED (mechanism goes unresolved). Never raises —
        this is the bookkeeping half of a refusal the caller already decided on."""
        self.inputs.append(CausalNumericInput(
            name=name, value=value, units=units, causal_role=causal_role,
            source_class=source_class, consumer=consumer, production_eligible=False,
            rejection_reason=why or f"source_class {source_class!r} not approved", consumed=False))

    # -------------------------------------------------------------- reporting
    def manifest(self) -> dict:
        out = {"approved_and_consumed": [], "approved_unused": [], "rejected": [],
               "compute_safety_only": [], "analysis_display_only": [], "ablation_only": []}
        for e in self.inputs:
            d = e.as_dict()
            if e.source_class == "compute_safety":
                out["compute_safety_only"].append(d)
            elif e.source_class in ("analysis_display", "numerical_tolerance"):
                out["analysis_display_only"].append(d)
            elif e.source_class == "ablation_only":
                out["ablation_only"].append(d)
            elif not e.production_eligible:
                out["rejected"].append(d)
            elif e.consumed:
                out["approved_and_consumed"].append(d)
            else:
                out["approved_unused"].append(d)
        out["n_inputs"] = len(self.inputs)
        out["n_rejected"] = len(out["rejected"])
        return out


# ------------------------------------------------------------------ ledger accessors
def ledger_of(world) -> NumericProvenanceLedger:
    """Get-or-create the WORLD's ledger (per branch). Worlds are plain objects; the ledger rides as
    an attribute so it needs no schema change and serializes only through the manifest."""
    led = getattr(world, "_numeric_ledger", None)
    if led is None:
        led = NumericProvenanceLedger()
        try:
            world._numeric_ledger = led
        except Exception:  # noqa: BLE001 — a world that refuses attributes still gets a ledger
            pass
    return led


def plan_ledger_of(plan) -> NumericProvenanceLedger:
    """Get-or-create the PLAN's ledger (conversion/compile-time decisions, before branches exist)."""
    led = getattr(plan, "_numeric_ledger", None)
    if led is None:
        led = NumericProvenanceLedger()
        try:
            plan._numeric_ledger = led
        except Exception:  # noqa: BLE001
            pass
    return led


def merge_manifests(*manifests) -> dict:
    """Merge per-branch/per-plan manifests into ONE result manifest, deduplicating identical
    (name, consumer, source_class, rejection_reason) rows and counting occurrences."""
    keys = ("approved_and_consumed", "approved_unused", "rejected", "compute_safety_only",
            "analysis_display_only", "ablation_only")
    seen, out = {}, {k: [] for k in keys}
    for m in manifests:
        if not isinstance(m, dict):
            continue
        for k in keys:
            for d in (m.get(k) or []):
                ident = (k, str(d.get("name")), str(d.get("consumer")),
                         str(d.get("source_class")), str(d.get("rejection_reason")))
                if ident in seen:
                    seen[ident]["n_occurrences"] += 1
                else:
                    row = dict(d, n_occurrences=1)
                    seen[ident] = row
                    out[k].append(row)
    out["n_inputs"] = sum(len(out[k]) for k in keys)
    out["n_rejected"] = len(out["rejected"])
    return out


# ------------------------------------------------------------------ fitted-artifact eligibility
#: every key a fitted parameter artifact must carry to be production-eligible
FITTED_ARTIFACT_REQUIRED_KEYS = (
    "version",              # versioned artifact id
    "training_population",  # what worlds/cases it was fit on
    "outcome_definition",   # the outcome semantics the parameters predict
    "data_cutoff",          # frozen cutoff — no leakage past it
    "n",                    # effective sample support
    "fitting_procedure",
    "heldout_metrics",      # held-out evaluation results
    "calibration",          # calibration diagnostics
    "domain_restrictions",  # where it applies
    "transport_check",      # evidence it transports to the target domain
    "architecture_version", # exact compatible architecture version
)
#: minimum effective sample support for a fitted artifact to serve production
FITTED_ARTIFACT_MIN_N = 50


def fitted_artifact_eligible(artifact: dict, *, architecture_version: str = "") -> tuple:
    """(eligible, why). An artifact is production-eligible only when it carries EVERY required
    provenance key with real content and sufficient support. A keyword-selected reference-class
    rate file with no held-out evaluation is NOT an empirically fitted mechanism."""
    if not isinstance(artifact, dict) or not artifact:
        return False, "no artifact supplied"
    missing = [k for k in FITTED_ARTIFACT_REQUIRED_KEYS if not artifact.get(k)]
    if missing:
        return False, f"missing required provenance keys: {', '.join(missing)}"
    try:
        n = float(artifact.get("n", 0))
    except (TypeError, ValueError):
        return False, "sample support `n` is not numeric"
    if n < FITTED_ARTIFACT_MIN_N:
        return False, f"insufficient effective sample support (n={artifact.get('n')} < " \
                      f"{FITTED_ARTIFACT_MIN_N})"
    if architecture_version and str(artifact.get("architecture_version")) != architecture_version:
        return False, (f"architecture version mismatch: artifact "
                       f"{artifact.get('architecture_version')!r} != runtime "
                       f"{architecture_version!r}")
    verdict = str(artifact.get("transport_check", "")).lower()
    if verdict in ("failed", "not_applicable", "none", "false"):
        return False, f"transport check verdict {artifact.get('transport_check')!r}"
    return True, "eligible"


# ------------------------------------------------------------------ unresolved-mechanism recording
def record_unresolved_mechanism(world, *, mechanism: str, why: str, missing: str = "",
                                var: str = "") -> dict:
    """The refusal side of the gate: a mechanism whose load-bearing number was rejected (or never
    existed) is recorded UNRESOLVED on the branch — the branch is preserved, the missing model is
    named, and no default is substituted. Terminal classification: a branch whose outcome-relevant
    mechanism carries an unresolved record must NOT read out as `resolved_no` — it is
    `unresolved_mechanism` mass (event_time honors this)."""
    rec = {"mechanism": str(mechanism)[:80], "why": str(why)[:240],
           "missing": str(missing)[:160], "outcome_var": str(var)[:80],
           "at_ts": getattr(getattr(world, "clock", None), "now", None),
           "classification": "unresolved_mechanism"}
    try:
        lst = getattr(world, "_unresolved_mechanisms", None)
        if lst is None:
            lst = []
            world._unresolved_mechanisms = lst
        lst.append(rec)
    except Exception:  # noqa: BLE001
        pass
    try:
        from swm.world_model_v2.temporal_runtime import get_stats
        stats = get_stats(world)
        if not hasattr(stats, "unresolved_mechanisms"):
            stats.unresolved_mechanisms = []
        stats.unresolved_mechanisms.append(rec)
    except Exception:  # noqa: BLE001
        pass
    try:
        world.omissions.append({"kind": "unresolved_mechanism", **rec})
    except Exception:  # noqa: BLE001
        pass
    return rec


def unresolved_mechanisms_of(world) -> list:
    """The branch's unresolved-mechanism records ([] when every consumed mechanism was modeled)."""
    return list(getattr(world, "_unresolved_mechanisms", None) or [])


def plan_record_unresolved(plan, *, mechanism: str, why: str, missing: str = "") -> dict:
    """Conversion-time refusal: recorded on the plan so every branch built from it inherits the
    unresolved record (materialize copies them onto worlds)."""
    rec = {"mechanism": str(mechanism)[:80], "why": str(why)[:240],
           "missing": str(missing)[:160], "classification": "unresolved_mechanism"}
    lst = getattr(plan, "_unresolved_mechanisms", None)
    if lst is None:
        lst = []
        plan._unresolved_mechanisms = lst
    lst.append(rec)
    return rec
