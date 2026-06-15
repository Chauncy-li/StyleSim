from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_NUPLAN_ROOT = Path(
    os.environ.get("NUPLAN_BASELINE_ROOT", "/home/lsw/meta_programs/Nuplan-Baseline-3090")
).expanduser()


def _normalize_repo_root(path: Optional[str]) -> Path:
    root = Path(path).expanduser() if path else DEFAULT_NUPLAN_ROOT
    if root.name == "nuplan_baseline":
        return root.parent.resolve()
    return root.resolve()


def resolve_nuplan_roots(
    nuplan_root: Optional[str] = None,
    nuplan_devkit_root: Optional[str] = None,
) -> Tuple[Path, Optional[Path]]:
    repo_root = _normalize_repo_root(nuplan_root)
    devkit_root: Optional[Path]
    if nuplan_devkit_root:
        devkit_root = Path(nuplan_devkit_root).expanduser().resolve()
    else:
        candidate = repo_root / "nuplan-devkit"
        devkit_root = candidate.resolve() if candidate.exists() else None
    return repo_root, devkit_root


def ensure_nuplan_imports(
    nuplan_root: Optional[str] = None,
    nuplan_devkit_root: Optional[str] = None,
) -> Tuple[Path, Optional[Path]]:
    repo_root, devkit_root = resolve_nuplan_roots(nuplan_root, nuplan_devkit_root)

    extra_paths = [repo_root]
    if devkit_root is not None:
        extra_paths.append(devkit_root)

    for path in extra_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return repo_root, devkit_root
