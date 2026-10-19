from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


SELECTED_CLASSES: List[str] = [
    "like",
    "dislike",
    "mute",
    "ok",
    "peace",
    "rock",
    "three_gun",
    "middle_finger",
    "hand_heart",
    "no_gesture",
]

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(SELECTED_CLASSES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for name, idx in CLASS_TO_ID.items()}

# Per-class confidence thresholds for detection + smoothing.
CLASS_THRESHOLDS: Dict[str, float] = {
    "like": 0.45,
    "dislike": 0.45,
    "mute": 0.40,
    "ok": 0.40,
    "peace": 0.40,
    "rock": 0.45,
    "three_gun": 0.50,
    "middle_finger": 0.55,
    "hand_heart": 0.45,
    "no_gesture": 0.35,
}


@dataclass
class PathsConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_root: Path = field(init=False)
    raw_root: Path = field(init=False)
    subset_root: Path = field(init=False)
    yolo_root: Path = field(init=False)
    dataset_yaml: Path = field(init=False)
    runs_root: Path = field(init=False)
    default_model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = self.project_root / "data"
        self.raw_root = self.data_root / "raw"
        self.subset_root = self.data_root / "hagrid_subset"
        self.yolo_root = self.data_root / "yolo"
        self.dataset_yaml = self.data_root / "hagrid_gestures.yaml"
        self.runs_root = self.project_root / "runs"
        self.default_model_path = self.project_root / "yolov13s.pt"


@dataclass
class InferenceConfig:
    webcam_index: int = 0
    input_size: int = 640
    effect_ignore_class: str = "no_gesture"
    smoothing_window_size: int = 10
    stable_frames_required: int = 6
    trigger_cooldown_sec: float = 1.0
    fps_smoothing_alpha: float = 0.9
    model_path: str = "auto"


@dataclass
class TrainingConfig:
    default_batch_size: int = 8
    default_imgsz: int = 640
    sanity_max_per_class: int = 100
    stage2_max_per_class: int = 1000
    stage3_max_per_class: int = 3000


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selected_classes: List[str] = field(default_factory=lambda: SELECTED_CLASSES.copy())
    class_to_id: Dict[str, int] = field(default_factory=lambda: CLASS_TO_ID.copy())
    id_to_class: Dict[int, str] = field(default_factory=lambda: ID_TO_CLASS.copy())
    class_thresholds: Dict[str, float] = field(default_factory=lambda: CLASS_THRESHOLDS.copy())


DEFAULT_CONFIG = AppConfig()
