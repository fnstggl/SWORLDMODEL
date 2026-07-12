"""Production mechanism registry — Phase 6.

Three layers: universal mechanism FAMILY → domain PARAMETER PACK → scenario instantiation (compiler).
Machine-readable state lives in registry/data/{registry,packs}.json (committed, integrity-hashed).
Lifecycle: proposed → implemented → locally_validated → transfer_validated → production_eligible,
with quarantined/rejected demotions — enforced by promotion gates, not labels.

    from swm.world_model_v2.registry import load_registry
    store = load_registry()          # loads committed records + mirrors into the compiler vocabulary
    store.summary()
"""
from swm.world_model_v2.registry.record import (ApplicabilityRule, Citation, MechanismRecord,
                                                ParameterPack, ParameterSpec, RegistryError,
                                                ValidationRecord)
from swm.world_model_v2.registry.store import RegistryStore
from swm.world_model_v2.registry.applicability import rank_mechanisms, score_applicability

_STORE = None


def load_registry(*, reload: bool = False) -> RegistryStore:
    global _STORE
    if _STORE is None or reload:
        _STORE = RegistryStore.load()
    return _STORE
