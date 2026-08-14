"""Fail if pyproject.toml's package list disagrees with the tree.

The distribution spans two source roots and most of kernels/ has no
__init__.py, so the packages are listed by hand rather than auto-discovered
(see the comment in pyproject.toml). A hand-written list rots: someone adds
kernels/foo/, the wheel silently omits it, and `import forge` fails only for
people who installed rather than cloned — which is how the wheel came to ship
no kernels at all in the first place.

Runs on CPU with no dependencies beyond the standard library.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Where each top-level package lives, mirroring [tool.setuptools] package-dir.
ROOTS = {"forge": Path("forge/forge"), "kernels": Path("kernels")}


def is_identifier(part: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is not None


def discover() -> set[str]:
    """Every directory holding a .py file that Python could import as a package.

    Directories whose names are not valid identifiers — the `01_unsloth` study
    copies under the knowledge bases — cannot be imported and are skipped.
    """
    found: set[str] = set()
    for top, root in ROOTS.items():
        base = REPO / root
        if not base.is_dir():
            continue
        for module in base.rglob("*.py"):
            rel = module.parent.relative_to(base)
            parts = [top] if str(rel) == "." else [top, *rel.parts]
            if not all(is_identifier(part) for part in parts):
                continue
            for depth in range(1, len(parts) + 1):
                found.add(".".join(parts[:depth]))
    return found


def declared() -> set[str]:
    with (REPO / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return set(config["tool"]["setuptools"]["packages"])


def main() -> int:
    on_disk, in_config = discover(), declared()

    missing = sorted(on_disk - in_config)
    stale = sorted(in_config - on_disk)

    for name in missing:
        print(f"::error::package {name} exists in the tree but is not listed in "
              f"pyproject.toml, so it would be left out of the wheel")
    for name in stale:
        print(f"::error::package {name} is listed in pyproject.toml but has no "
              f".py files in the tree")

    if missing or stale:
        print(f"\n{len(missing)} missing, {len(stale)} stale. "
              f"Update [tool.setuptools] packages in pyproject.toml.")
        return 1

    print(f"pyproject.toml declares all {len(on_disk)} importable packages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
