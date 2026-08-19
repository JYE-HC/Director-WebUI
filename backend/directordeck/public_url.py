"""Public URL prefix for browser-facing API URLs embedded in responses.

The backend runs embedded behind the ComfyUI plugin's reverse-proxy mount
point (``/directordeck/api``); the host calls ``set_public_api_prefix`` so
response URLs point at the public mount instead of colliding with the host's
own ``/api`` namespace.

The prefix is process-global because the single-instance database lock
already guarantees at most one Director app per process.
"""

from __future__ import annotations

_prefix = ""


def set_public_api_prefix(prefix: str) -> None:
    global _prefix
    _prefix = prefix.rstrip("/")


def public_api_url(path: str) -> str:
    return f"{_prefix}{path}"
