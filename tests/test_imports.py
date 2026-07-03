"""Tests that every module under ``src/synlad/`` is importable.

Modules that depend on optional licensed packages (OpenEye) are skipped
automatically when those packages are not installed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

OPTIONAL_DEPS = ("openeye",)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "synlad"


def _discover_modules() -> list[str]:
    modules: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


MODULE_NAMES = _discover_modules()


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in OPTIONAL_DEPS:
            pytest.skip(f"optional dependency {missing!r} not installed")
        raise
