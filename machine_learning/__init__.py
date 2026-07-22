"""SWORLDMODEL behaviour-ML system.

A self-contained, auditable pipeline for acquiring, normalizing, validating, splitting,
sampling, and training on real human-behaviour datasets. Fully isolated from the
production ``swm`` runtime: this package never imports ``swm`` and ``swm`` never imports
this package.

Sub-packages
------------
* ``acquisition``  — download + verify source datasets (resumable, disk-safe)
* ``normalization``— convert raw records into the canonical behaviour-event schema
* ``examples``     — build task-specific training examples + format them for SFT
* ``splitting``    — leakage-safe split policies + leakage checks
* ``sampling``     — balanced training-view manifests
* ``validation``   — schema / chronology / leakage / dedup / provenance / licensing checks
* ``training``     — GPU-ready QLoRA training pipeline (also a CPU smoke path)
* ``evaluation``   — per-task-family metrics + baselines
* ``cli``          — the ``python -m machine_learning.cli`` entrypoint
"""
from __future__ import annotations

__version__ = "0.1.0"
