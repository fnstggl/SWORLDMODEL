"""Converter base class + dynamic loader.

A converter turns one dataset's raw records into canonical behaviour-event records. It
implements :meth:`iter_records` (a generator over canonical records built with
:meth:`make`) and declares a :attr:`DOC` describing exactly how the real source maps to
the canonical schema. It never fabricates data: absent source fields become
null/[]/{} and are listed in ``data_quality.missing_fields``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterator

from ..canonical import make_record
from .common.pseudonymize import pseudonym


class Converter:
    DATASET_ID: str = ""
    VERSION: str = "1.0.0"
    #: Converter documentation, validated against schemas/source_manifests/converter_doc.schema.json
    DOC: dict = {}
    #: Optional fixtures subdir name under tests/fixtures/ used by the fixture self-test.
    FIXTURE_SUBDIR: str | None = None

    def __init__(self, entry: dict | None = None, *, timestamp: str | None = None,
                 code_commit: str | None = None):
        from ..registry_io import get_dataset
        self.entry = entry if entry is not None else get_dataset(self.DATASET_ID)
        self.license_class = self.entry.get("license_class", "unknown_unstated")
        self.timestamp = timestamp
        self.code_commit = code_commit

    # ---- identity ----------------------------------------------------------------------
    @property
    def converter_path(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}"

    def pseudonym(self, kind: str, raw_id) -> str:
        return pseudonym(self.DATASET_ID, kind, raw_id)

    # ---- record construction -----------------------------------------------------------
    def make(self, *, task_type: str, payload: dict, episode_id: str,
             raw_locator: dict, transformation_steps: list[str] | None = None, **kw) -> dict:
        """Build a canonical record with this converter's dataset/version/provenance defaults."""
        return make_record(
            dataset_id=self.DATASET_ID,
            task_type=task_type,
            payload=payload,
            converter=self.converter_path,
            converter_version=self.VERSION,
            license_class=self.license_class,
            citation=self.entry.get("official_paper", ""),
            normalization_timestamp=self.timestamp,
            code_commit=self.code_commit,
            episode_id=episode_id,
            raw_locator=raw_locator,
            transformation_steps=transformation_steps or [],
            **kw,
        )

    # ---- the one thing subclasses must implement ---------------------------------------
    def iter_records(self, raw_dir: Path) -> Iterator[dict]:
        raise NotImplementedError(f"{self.converter_path} must implement iter_records()")


def load_converter(dataset_id: str, **kw) -> Converter | None:
    """Instantiate the converter registered for ``dataset_id`` (or None if none)."""
    from ..registry_io import get_dataset
    entry = get_dataset(dataset_id)
    path = entry.get("converter")
    if not path:
        return None
    module_path, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls(entry=entry, **kw)


class SourceNotAvailable(RuntimeError):
    """Raised by a converter when its raw source is absent/unreleased."""
