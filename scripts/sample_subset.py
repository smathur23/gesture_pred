from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from scripts.common import ensure_dir, find_image_path, iter_annotation_items, pick_bboxes_for_class

DEFAULT_CLASSES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create class-balanced manifest subsets from HaGRID annotations.")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to hagrid_annotations directory")
    parser.add_argument("--images", type=Path, required=True, help="Path to hagrid_dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL manifest path")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-image-exists", action="store_true")
    return parser.parse_args()


def reservoir_sample_rows(
    annotation_path: Path,
    split: str,
    class_name: str,
    images_root: Path,
    max_samples: int,
    rng: random.Random,
    require_image_exists: bool,
) -> List[dict]:
    reservoir: List[dict] = []
    seen = 0
    for image_id, row in iter_annotation_items(annotation_path):
        selected_boxes = pick_bboxes_for_class(row, class_name)
        if not selected_boxes:
            continue

        image_path = find_image_path(images_root, class_name, image_id)
        if require_image_exists and image_path is None:
            continue

        candidate = {
            "split": split,
            "class_name": class_name,
            "image_id": image_id,
            "image_exists": bool(image_path is not None),
            "annotation_file": str(annotation_path),
        }
        seen += 1
        if len(reservoir) < max_samples:
            reservoir.append(candidate)
            continue
        j = rng.randint(0, seen - 1)
        if j < max_samples:
            reservoir[j] = candidate
    return reservoir


def build_manifest(args: argparse.Namespace) -> Iterable[dict]:
    rng = random.Random(args.seed)
    for split in args.splits:
        split_dir = args.annotations / split
        if not split_dir.exists():
            continue
        for class_name in args.classes:
            annotation_path = split_dir / f"{class_name}.json"
            if not annotation_path.exists():
                continue
            sampled_rows = reservoir_sample_rows(
                annotation_path=annotation_path,
                split=split,
                class_name=class_name,
                images_root=args.images,
                max_samples=args.max_per_class,
                rng=rng,
                require_image_exists=args.require_image_exists,
            )
            for row in sampled_rows:
                yield row


def main() -> None:
    args = parse_args()
    ensure_dir(args.out.parent)
    stats: Dict[str, int] = {}
    total = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for row in tqdm(build_manifest(args), desc="Sampling", unit="img"):
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            key = f'{row["split"]}:{row["class_name"]}'
            stats[key] = stats.get(key, 0) + 1
            total += 1

    print(f"Saved {total} rows to {args.out}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
