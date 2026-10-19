from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from app.model_resolver import find_latest_weight, resolve_model_path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pt")


class TestModelResolver(unittest.TestCase):
    def test_find_latest_best_prefers_newer_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "detect" / "run_a" / "weights" / "best.pt"
            p2 = root / "detect" / "run_b" / "weights" / "best.pt"
            _touch(p1)
            _touch(p2)
            os.utime(p1, (1000, 1000))
            os.utime(p2, (2000, 2000))
            latest = find_latest_weight(root, preference="best")
            self.assertIsNotNone(latest)
            self.assertEqual(latest, p2)

    def test_resolve_auto_uses_latest_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runs_root = project_root / "runs"
            best = runs_root / "detect" / "exp1" / "weights" / "best.pt"
            _touch(best)
            resolved = resolve_model_path("auto", project_root=project_root, runs_root=runs_root)
            self.assertEqual(resolved, best.resolve())

    def test_resolve_default_name_falls_back_to_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            runs_root = project_root / "runs"
            best = runs_root / "detect" / "exp_fallback" / "weights" / "best.pt"
            _touch(best)
            resolved = resolve_model_path("yolov13s.pt", project_root=project_root, runs_root=runs_root)
            self.assertEqual(resolved, best.resolve())

    def test_missing_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_model_path("missing.pt", project_root=root, runs_root=root / "runs")


if __name__ == "__main__":
    unittest.main()
