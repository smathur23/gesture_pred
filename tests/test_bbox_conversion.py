import unittest

from scripts.common import coco_xywh_to_yolo_xywh


class TestBBoxConversion(unittest.TestCase):
    def test_basic_conversion(self) -> None:
        cx, cy, w, h = coco_xywh_to_yolo_xywh((0.2, 0.3, 0.4, 0.5))
        self.assertAlmostEqual(cx, 0.4, places=6)
        self.assertAlmostEqual(cy, 0.55, places=6)
        self.assertAlmostEqual(w, 0.4, places=6)
        self.assertAlmostEqual(h, 0.5, places=6)

    def test_clipping(self) -> None:
        cx, cy, w, h = coco_xywh_to_yolo_xywh((-0.2, 0.95, 1.4, 0.3))
        self.assertGreaterEqual(cx, 0.0)
        self.assertLessEqual(cx, 1.0)
        self.assertGreaterEqual(cy, 0.0)
        self.assertLessEqual(cy, 1.0)
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0)


if __name__ == "__main__":
    unittest.main()

