import unittest

from app.detector import Detection
from app.gesture_smoother import GestureSmoother


class TestSmoother(unittest.TestCase):
    def test_triggers_only_after_stable_frames(self) -> None:
        smoother = GestureSmoother(
            per_class_thresholds={"like": 0.4, "no_gesture": 0.1},
            window_size=6,
            stable_frames_required=4,
            cooldown_sec=1.0,
            ignored_trigger_class="no_gesture",
        )
        det = Detection(0, "like", 0.9, (10, 10, 20, 20))

        states = []
        for i in range(4):
            states.append(smoother.update(det, now=float(i)))

        self.assertFalse(states[0].should_trigger)
        self.assertFalse(states[1].should_trigger)
        self.assertFalse(states[2].should_trigger)
        self.assertTrue(states[3].should_trigger)

    def test_cooldown_blocks_retrigger(self) -> None:
        smoother = GestureSmoother(
            per_class_thresholds={"like": 0.4, "no_gesture": 0.1},
            window_size=6,
            stable_frames_required=3,
            cooldown_sec=2.0,
            ignored_trigger_class="no_gesture",
        )
        det = Detection(0, "like", 0.95, (0, 0, 1, 1))
        smoother.update(det, now=0.0)
        smoother.update(det, now=0.1)
        first = smoother.update(det, now=0.2)
        second = smoother.update(det, now=0.3)
        self.assertTrue(first.should_trigger)
        self.assertFalse(second.should_trigger)

    def test_no_gesture_never_triggers(self) -> None:
        smoother = GestureSmoother(
            per_class_thresholds={"no_gesture": 0.1},
            window_size=5,
            stable_frames_required=3,
            cooldown_sec=0.1,
            ignored_trigger_class="no_gesture",
        )
        s1 = smoother.update(None, now=0.0)
        s2 = smoother.update(None, now=0.1)
        s3 = smoother.update(None, now=0.2)
        self.assertFalse(s1.should_trigger)
        self.assertFalse(s2.should_trigger)
        self.assertFalse(s3.should_trigger)
        self.assertEqual(s3.stable_gesture, "no_gesture")


if __name__ == "__main__":
    unittest.main()

