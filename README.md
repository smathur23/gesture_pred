# Gesture Effects MVP (YOLOv13 + HaGRIDv2)

Real-time webcam hand-gesture visual effects MVP inspired by FaceTime gesture reactions.

This project:
- Uses a 10-class HaGRIDv2 subset.
- Converts annotations to YOLO format.
- Trains YOLOv13-S.
- Runs webcam inference with smoothing, cooldown, and OpenCV visual effects.

## Selected Classes

Class order is fixed and used by dataset + model labels:

1. `like`
2. `dislike`
3. `mute`
4. `ok`
5. `peace`
6. `rock`
7. `three_gun`
8. `middle_finger`
9. `hand_heart`
10. `no_gesture`

## Project Structure

```text
gesture_pred/
├── data/
│   ├── raw/
│   ├── hagrid_subset/
│   ├── yolo/
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── hagrid_gestures.yaml
├── scripts/
├── app/
│   └── effects/
├── assets/
├── runs/
├── tests/
├── requirements.txt
└── README.md
```

## Environment Setup (Conda `gestvenv`)

```powershell
conda activate gestvenv
python --version
```

Clone upstream repos:

```powershell
mkdir vendor
git clone https://github.com/iMoonLab/yolov13 vendor/yolov13
git clone https://github.com/hukenovs/hagrid vendor/hagrid
```

Install dependencies:

```powershell
pip install -r requirements.txt
pip install -e vendor/yolov13
```

Notes:
- If `pip install -e vendor/yolov13` fails on Windows due to Linux-only Flash Attention wheels, train in WSL2/Linux and keep webcam inference on Windows using exported or `.pt` weights.

## Data Preparation

Download selected classes + annotations using official HaGRID downloader:

```powershell
python scripts/download_hagrid_subset.py --annotations --images --classes like dislike mute ok peace rock three_gun middle_finger hand_heart no_gesture --out data/raw
```

Create a sanity subset manifest (100 per class):

```powershell
python scripts/sample_subset.py --annotations data/raw/hagrid_annotations --images data/raw/hagrid_dataset --out data/hagrid_subset/sanity_manifest.jsonl --max-per-class 100 --splits train val test --require-image-exists
```

Convert to YOLO layout (symlink/hardlink/copy fallback):

```powershell
python scripts/convert_hagrid_to_yolo.py --manifest data/hagrid_subset/sanity_manifest.jsonl --images data/raw/hagrid_dataset --out data/yolo --link-mode auto
```

Visualize labels:

```powershell
python scripts/visualize_labels.py --data data/hagrid_gestures.yaml --split train --samples 40 --out runs/label_viz
```

## Training Stages

Stage 1 (sanity check):

```powershell
python scripts/train_yolov13.py --data data/hagrid_gestures.yaml --model yolov13s.pt --epochs 5 --imgsz 640 --batch 8 --device 0 --name sanity_100
```

Stage 2 (small experiment ~1000/class):

```powershell
python scripts/sample_subset.py --annotations data/raw/hagrid_annotations --images data/raw/hagrid_dataset --out data/hagrid_subset/small_manifest.jsonl --max-per-class 1000 --splits train val test --require-image-exists
python scripts/convert_hagrid_to_yolo.py --manifest data/hagrid_subset/small_manifest.jsonl --images data/raw/hagrid_dataset --out data/yolo --link-mode auto --clean
python scripts/train_yolov13.py --data data/hagrid_gestures.yaml --model yolov13s.pt --epochs 30 --imgsz 640 --batch 8 --device 0 --name small_1000
```

Stage 3 (MVP model ~1500-3000/class):

```powershell
python scripts/sample_subset.py --annotations data/raw/hagrid_annotations --images data/raw/hagrid_dataset --out data/hagrid_subset/final_manifest.jsonl --max-per-class 3000 --splits train val test --require-image-exists
python scripts/convert_hagrid_to_yolo.py --manifest data/hagrid_subset/final_manifest.jsonl --images data/raw/hagrid_dataset --out data/yolo --link-mode auto --clean
python scripts/train_yolov13.py --data data/hagrid_gestures.yaml --model yolov13s.pt --epochs 80 --imgsz 640 --batch 8 --device 0 --name mvp_3000
```

## Webcam Inference + Effects

Run app:

```powershell
python app/main.py --model auto
```

Use an explicit checkpoint:

```powershell
python app/main.py --model runs/detect/mvp_3000/weights/best.pt
```

Use webcam + mirror + controls overlay:

```powershell
python app/main.py --model auto --source 0 --mirror --show-controls
```

Check visible webcam indices:

```powershell
python app/main.py --list-cameras
```

Record the processed output:

```powershell
python app/main.py --model auto --source 0 --record-out runs/webcam/session1.mp4
```

Controls:
- Press `q` to quit.
- Press `b` to toggle detection boxes.
- Press `e` to toggle gesture effects.
- Press `m` to toggle mirror mode.
- Press `h` to toggle controls overlay.

WSL note:
- If webcam fails in WSL2 with camera index errors, run the app from native Windows Python/Conda for webcam access, or provide a video file via `--source path/to/video.mp4`.

Display includes:
- Bounding boxes
- Gesture labels/confidence
- Dominant + stable gesture
- FPS

## Benchmark + Export

```powershell
python scripts/benchmark_webcam.py --model runs/detect/mvp_3000/weights/best.pt --seconds 60
python scripts/export_model.py --weights runs/detect/mvp_3000/weights/best.pt --format onnx
```

## Memory/Storage Strategy

- Annotation handling is streaming (`ijson` when available), not image-array loading.
- Subset creation uses balanced per-class limits.
- Image placement uses `symlink -> hardlink -> copy` fallback (`--link-mode auto`).
- Split preservation follows `hagrid_annotations/{train,val,test}`.
- Keep full raw archives in `data/raw`, and only keep selected subset under `data/yolo`.

## Tests

Run unit tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Covered:
- bbox conversion (`COCO xywh -> YOLO cxcywh`)
- sampling constraints
- smoothing stability/cooldown/no-gesture suppression
- automatic model checkpoint selection
