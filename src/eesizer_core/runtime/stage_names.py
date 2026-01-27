from __future__ import annotations

import re


_STAGE_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_UNDERSCORE_RUN_RE = re.compile(r"_+")


def sanitize_stage_name(value: str, *, default: str = "stage") -> str:
    """Return a filesystem-friendly stage name that matches [A-Za-z0-9_-]+.

    This is used for ngspice stage directory names (ngspice_runner enforces a strict
    regex), so we sanitize instead of failing late.
    """
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = _STAGE_SAFE_RE.sub("_", text)
    text = _UNDERSCORE_RUN_RE.sub("_", text).strip("_")
    return text or default

