"""Quarantined experimental modules — NOT imported by the production front door.

Anything here is unvalidated and/or license-restricted (see the per-module docstrings and docs/AUDIT_*.md).
The agent engine (`swm/engine/*`) must never import from this package; a test pins that. These exist so the
interface/experiment can be developed and reviewed without any risk of an unvalidated component silently
entering a forecast.
"""
