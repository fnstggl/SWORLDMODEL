"""Ephemeral credential loading for benchmark commands.

The secret is read into process memory from a user-owned source file.  It is
never returned in diagnostics, written to artifacts, placed in argv, or copied
to the environment.
"""
from __future__ import annotations

import re
from pathlib import Path


_MARKER = re.compile(r"(?:DEEPSEEK_API_KEY\s*=\s*)?(sk-[A-Za-z0-9]{20,})")


def read_deepseek_key(source: Path) -> str:
    text = source.read_text()
    match = _MARKER.search(text)
    if not match:
        raise ValueError("DeepSeek credential marker not found in the supplied user-owned source")
    key = match.group(1)
    if not key.isascii():
        raise ValueError("DeepSeek credential must contain only ASCII characters")
    return key
