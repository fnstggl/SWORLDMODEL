"""AST / source enforcement: the scenario-generated action layer may never resurrect the
legacy global verb catalog, and the generated-world kernel must stay semantically empty.

These are static guards — they read production source with the ``ast`` module and never run it,
so a future edit that smuggles the fixed Phase-13 operation registry back into the generated
path (an import, an attribute access, or a resurrected module-level verb whitelist) fails here.
"""
import ast
import pathlib

import pytest

import swm.world_model_v2.phase13.scenario_actions as pkg
from swm.world_model_v2.generated_world import KERNEL_OPS
from swm.world_model_v2.phase13 import ontology

PKG_DIR = pathlib.Path(pkg.__file__).parent
MODULE_FILES = sorted(p for p in PKG_DIR.glob("*.py"))

#: names that would re-couple the generated path to the fixed operation catalog
FORBIDDEN_ONTOLOGY_NAMES = {"_OPERATIONS", "operation_registered", "operation_spec",
                            "OPERATION_FAMILIES"}
#: the legacy verb set — imported ONLY to build the reference for the whitelist check
LEGACY_VERBS = set(ontology._OPERATIONS)

#: the kernel is storage/integrity mechanics only — this is the entire permitted surface.
#: invoke_scenario_mechanism is semantically empty by the same standard: it names a
#: SCENARIO-generated mechanism id exactly as emit_semantic_event names a scenario event
#: type; the kernel only validates shape/authority/preconditions and starts the instance.
KERNEL_ALLOWLIST = {"declare_schema_definition", "create_or_update_record", "remove_record",
                    "create_or_remove_relation", "emit_semantic_event", "schedule_semantic_event",
                    "transfer_conserved_quantity", "invoke_scenario_mechanism"}


def test_the_package_has_modules_to_check():
    names = {p.name for p in MODULE_FILES}
    assert {"api.py", "compiler.py", "execution.py", "language.py", "generated_search.py"} <= names


def _ontology_aliases(tree) -> set:
    """Local names bound to the legacy ontology module (import forms)."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("phase13") or mod.endswith("world_model_v2.phase13"):
                for a in node.names:
                    if a.name == "ontology":
                        aliases.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith(".ontology") or a.name == "ontology":
                    aliases.add(a.asname or a.name.split(".")[-1])
    return aliases


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda p: p.name)
def test_module_never_imports_the_legacy_operation_registry(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    # (a1) direct `from ...ontology import <forbidden>`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("ontology"):
            leaked = {a.name for a in node.names} & FORBIDDEN_ONTOLOGY_NAMES
            assert not leaked, f"{path.name} imports {leaked} from the legacy ontology"
    # (a2) attribute access `ontology.<forbidden>` on any ontology alias
    aliases = _ontology_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in aliases:
            assert node.attr not in FORBIDDEN_ONTOLOGY_NAMES, \
                f"{path.name} reaches {node.value.id}.{node.attr} on the legacy ontology"


def _string_constants(node) -> list:
    """String constants in a collection literal (list/tuple/set elements or dict keys)."""
    out = []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elts = node.elts
    elif isinstance(node, ast.Dict):
        elts = node.keys
    else:
        return out
    for e in elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            out.append(e.value)
    return out


@pytest.mark.parametrize("path", MODULE_FILES, ids=lambda p: p.name)
def test_module_has_no_resurrected_global_verb_whitelist(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        strings = _string_constants(node)
        if len(strings) < 10:
            continue
        overlap = set(strings) & LEGACY_VERBS
        assert len(overlap) < 10, (
            f"{path.name} declares a {len(strings)}-string literal overlapping the legacy verb "
            f"catalog in {len(overlap)} names {sorted(overlap)[:12]} — a resurrected global "
            f"verb whitelist")


def test_kernel_stays_semantically_empty():
    assert len(KERNEL_OPS) <= 8, f"the kernel grew to {len(KERNEL_OPS)} ops — it must stay small"
    extra = set(KERNEL_OPS) - KERNEL_ALLOWLIST
    assert not extra, f"kernel gained non-storage ops {extra} — meanings must be scenario-generated"
    # sanity: none of the kernel ops is a legacy verb name (they are storage mechanics)
    assert not (set(KERNEL_OPS) & LEGACY_VERBS)
