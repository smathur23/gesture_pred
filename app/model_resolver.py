from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_latest_weight(runs_root: Path, preference: str = "best") -> Optional[Path]:
    detect_root = runs_root / "detect"
    if not detect_root.exists():
        return None

    candidates: list[Path] = []
    for run_dir in detect_root.iterdir():
        if not run_dir.is_dir():
            continue
        weights = run_dir / "weights"
        if not weights.exists():
            continue
        if preference in {"best", "auto"}:
            best = weights / "best.pt"
            if best.exists():
                candidates.append(best)
        if preference in {"last", "auto"}:
            last = weights / "last.pt"
            if last.exists():
                candidates.append(last)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_model_path(requested_model: str, project_root: Path, runs_root: Path, default_name: str = "yolov13s.pt") -> Path:
    requested = Path(requested_model).expanduser()
    if requested.exists():
        return requested.resolve()

    model_alias = requested_model.strip().lower()
    if model_alias in {"auto", "latest", "latest-best"}:
        latest = find_latest_weight(runs_root, preference="best")
        if latest is not None:
            return latest.resolve()
    if model_alias == "latest-last":
        latest = find_latest_weight(runs_root, preference="last")
        if latest is not None:
            return latest.resolve()

    # Compatibility fallback: if default model is not present, try latest best run.
    if requested_model == default_name:
        fallback = find_latest_weight(runs_root, preference="best")
        if fallback is not None:
            return fallback.resolve()

    # If path is relative, resolve against project root for clearer diagnostics.
    if not requested.is_absolute():
        requested = (project_root / requested).resolve()
    raise FileNotFoundError(
        f"Model not found: {requested}. "
        "Provide --model with a valid .pt path, or use --model auto to load the latest best checkpoint."
    )
