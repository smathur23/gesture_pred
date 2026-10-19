from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm

try:
    from scripts.common import (
        ensure_dir,
        find_image_path,
        iter_annotation_items,
        link_or_copy,
        pick_bboxes_for_class,
        safe_filename,
    )
except ModuleNotFoundError:
    from common import (
        ensure_dir,
        find_image_path,
        iter_annotation_items,
        link_or_copy,
        pick_bboxes_for_class,
        safe_filename,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sampled HaGRID subset to YOLO dataset format.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL from sample_subset.py")
    parser.add_argument("--images", type=Path, required=True, help="Path to hagrid_dataset")
    parser.add_argument("--out", type=Path, default=Path("data/yolo"), help="YOLO dataset root")
    parser.add_argument("--link-mode", choices=["auto", "symlink", "hardlink", "copy"], default="auto")
    parser.add_argument("--clean", action="store_true", help="Clear images/labels under output root first")
    return parser.parse_args()


def maybe_clean(out_root: Path) -> None:
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        p = out_root / sub
        if p.exists():
            shutil.rmtree(p)


def ensure_layout(out_root: Path) -> None:
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        ensure_dir(out_root / sub)


def load_manifest_index(manifest_path: Path) -> tuple[Dict[tuple[str, str], Set[str]], Dict[tuple[str, str], Path]]:
    wanted: Dict[tuple[str, str], Set[str]] = {}
    annotation_map: Dict[tuple[str, str], Path] = {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = row["split"]
            class_name = row["class_name"]
            image_id = str(row["image_id"])
            key = (split, class_name)
            wanted.setdefault(key, set()).add(image_id)
            if "annotation_file" in row and row["annotation_file"]:
                annotation_map.setdefault(key, Path(row["annotation_file"]))
    return wanted, annotation_map


def convert(args: argparse.Namespace) -> None:
    if args.clean:
        maybe_clean(args.out)
    ensure_layout(args.out)

    selected_ids, annotation_map = load_manifest_index(args.manifest)
    total_written = 0
    link_stats: Dict[str, int] = {}
    class_to_id = {name: idx for idx, name in enumerate(class_order())}

    for (split, class_name), image_ids in sorted(selected_ids.items()):
        annotation_path = annotation_map.get((split, class_name), Path("data/raw/hagrid_annotations") / split / f"{class_name}.json")

        if not annotation_path.exists():
            print(f"Warning: missing annotation file {annotation_path}")
            continue

        out_image_dir = args.out / "images" / split
        out_label_dir = args.out / "labels" / split
        ensure_dir(out_image_dir)
        ensure_dir(out_label_dir)

        for image_id, row in tqdm(iter_annotation_items(annotation_path), desc=f"{split}:{class_name}", unit="img"):
            if image_id not in image_ids:
                continue
            yolo_boxes = pick_bboxes_for_class(row, class_name)
            if not yolo_boxes:
                continue
            src_image = find_image_path(args.images, class_name, image_id)
            if src_image is None:
                continue

            out_image_name = safe_filename(class_name, image_id, src_image.suffix.lower())
            out_label_name = safe_filename(class_name, image_id, ".txt")
            dst_image = out_image_dir / out_image_name
            dst_label = out_label_dir / out_label_name

            mode_used = link_or_copy(src_image, dst_image, mode=args.link_mode)
            link_stats[mode_used] = link_stats.get(mode_used, 0) + 1

            with dst_label.open("w", encoding="utf-8") as fh:
                for _, (cx, cy, w, h) in yolo_boxes:
                    class_id = class_to_id[class_name]
                    fh.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            total_written += 1

    print(f"Converted {total_written} images to YOLO format at {args.out}")
    print("Placement mode counts:", link_stats)


def class_order() -> List[str]:
    return [
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


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
