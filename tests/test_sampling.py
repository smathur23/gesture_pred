import json
import tempfile
import unittest
from pathlib import Path

from scripts.sample_subset import reservoir_sample_rows


class TestSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.annotations = root / "hagrid_annotations" / "train"
        self.images = root / "hagrid_dataset" / "like"
        self.annotations.mkdir(parents=True, exist_ok=True)
        self.images.mkdir(parents=True, exist_ok=True)

        data = {}
        for i in range(20):
            image_id = f"{i:08d}"
            data[image_id] = {
                "bboxes": [[0.1, 0.2, 0.3, 0.4]],
                "labels": ["like"],
                "united_bbox": [],
                "united_label": [],
            }
            if i % 2 == 0:
                (self.images / f"{image_id}.jpg").write_bytes(b"fake")
        (self.annotations / "like.json").write_text(json.dumps(data), encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_respects_max(self) -> None:
        rows = reservoir_sample_rows(
            annotation_path=self.annotations / "like.json",
            split="train",
            class_name="like",
            images_root=self.images.parent,
            max_samples=5,
            rng=__import__("random").Random(1),
            require_image_exists=False,
        )
        self.assertLessEqual(len(rows), 5)

    def test_require_image_exists(self) -> None:
        rows = reservoir_sample_rows(
            annotation_path=self.annotations / "like.json",
            split="train",
            class_name="like",
            images_root=self.images.parent,
            max_samples=20,
            rng=__import__("random").Random(2),
            require_image_exists=True,
        )
        self.assertTrue(rows)
        self.assertTrue(all(row["image_exists"] for row in rows))
        self.assertLessEqual(len(rows), 10)


if __name__ == "__main__":
    unittest.main()

