"""Phase 10 — production institutional world modeling.

Universal institution model (FAMILY → TEMPLATE → INSTANCE) with evidence-backed, temporally-versioned rules
that EXECUTE through the Phase-1 WorldState / StateDelta path, BLOCK unauthorized actions, and feed Phase-6
behavioral mechanisms operating inside institutional constraints. The compiler requests institutions BY
CAUSAL NEED (not keyword routing); Phase-3 posteriors carry institutional structural uncertainty.

    from swm.world_model_v2.institutions_v2 import load_store, select_institution
    store = load_store()
"""
from swm.world_model_v2.institutions_v2.store import InstitutionStore, load_store
from swm.world_model_v2.institutions_v2.record import (InstitutionFamily, InstitutionTemplate,
                                                       InstitutionInstance, EvidenceRecord, RuleRecord)

__all__ = ["InstitutionStore", "load_store", "InstitutionFamily", "InstitutionTemplate",
           "InstitutionInstance", "EvidenceRecord", "RuleRecord", "select_institution"]


def select_institution(*args, **kwargs):
    from swm.world_model_v2.institutions_v2.compile import select_institution as _sel
    return _sel(*args, **kwargs)
