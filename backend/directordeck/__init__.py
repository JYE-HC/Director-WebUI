"""Director Web backend.

Keep the package import light so workflow-only tools can run inside ComfyUI's
Python environment without importing the web application's dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import create_app as create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
