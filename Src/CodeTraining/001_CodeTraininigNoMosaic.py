#!/usr/bin/env python3
# 012b_TrainYOLOv8_RBCWBC_NO_MOSAIC.py
# Train YOLOv8 RBC/WBC WITHOUT mosaic (mosaic=0.0) + resume otomatis

from pathlib import Path
from ultralytics import YOLO

# =========================
# (WAJIB) PATH PROJECT
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (BASE_DIR / ".." / "..").resolve()

DATASETS_DIR = PROJECT_ROOT / "Data" / "Datasets"
MODELS_DIR = PROJECT_ROOT / "Data" / "DataModels"
RUNS_ROOT = MODELS_DIR / "runs"

# =========================
# (WAJIB) DATASET
# =========================
YOLO_DATASET_DIR = DATASETS_DIR / "RBC_WBC_YOLO_DATASET"
DATASET_YAML = YOLO_DATASET_DIR / "data.yaml"
if not DATASET_YAML.exists():
    raise FileNotFoundError(f"data.yaml tidak ditemukan: {DATASET_YAML}")

# =========================
# (WAJIB) TRAIN PARAMS
# =========================
model_name = "yolov8s.pt"
imgsz = 640
epochs = 60
batch = -1
device = 0
workers = 4

# =========================
# (OPSIONAL, recommended)
# =========================
seed = 42
deterministic = True

# (OPSIONAL) augmentasi selain mosaic (dibuat aman)
AUG_COMMON = dict(
    fliplr=0.5,
    flipud=0.0,
    translate=0.05,
    scale=0.20,
    shear=0.0,
    perspective=0.0,
    mixup=0.0,
    copy_paste=0.0,
)

# =========================
# MOSAIC SETTING (INI BEDANYA)
# =========================
MOSAIC = 0.0
RUN_SUFFIX = "_no_mosaic"


def main():
    dataset_name = YOLO_DATASET_DIR.name
    run_name = f"{dataset_name}_{imgsz}_{Path(model_name).stem}{RUN_SUFFIX}"
    run_dir = RUNS_ROOT / "detect" / run_name
    last_ckpt = run_dir / "weights" / "last.pt"

    # Resume otomatis
    if last_ckpt.exists():
        print("♻️ RESUME TRAINING")
        print(f"checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print("🆕 NEW TRAINING")
        print(f"pretrained: {model_name}")
        model = YOLO(model_name)

    print("\n--- RUN CONFIG ---")
    print(f"run_name : {run_name}")
    print(f"mosaic   : {MOSAIC}")
    print(f"imgsz    : {imgsz}")
    print(f"epochs   : {epochs}")
    print(f"batch    : {batch}")
    print(f"device   : {device}")
    print("------------------\n")

    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,

        project=str(RUNS_ROOT),
        name=run_name,

        patience=20,
        cos_lr=True,
        cache=False,

        seed=seed,
        deterministic=deterministic,

        mosaic=MOSAIC,
        **AUG_COMMON,
    )

    print("\n✅ DONE")
    print(f"📁 Output: {run_dir}")


if __name__ == "__main__":
    main()
