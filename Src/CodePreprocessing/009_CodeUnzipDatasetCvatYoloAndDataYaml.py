#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def infer_dataset_root(extract_dir: Path) -> Path:
    items = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return extract_dir


def move_all_files(src: Path, dst: Path, exts=None) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue

        target = dst / p.name
        if target.exists():
            stem, suf = target.stem, target.suffix
            i = 1
            while True:
                cand = dst / f"{stem}_{i}{suf}"
                if not cand.exists():
                    target = cand
                    break
                i += 1

        shutil.move(str(p), str(target))
        n += 1
    return n


def cleanup_empty_dirs(path: Path):
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def parse_names_from_yaml(yaml_path: Path):
    default = {0: "RBC", 1: "WBC"}
    if not yaml_path.exists():
        return default

    names = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if not line.startswith(" "):
                break
            if ":" in line:
                k, v = line.strip().split(":", 1)
                if k.isdigit():
                    names[int(k)] = v.strip()
    return names if names else default


def write_clean_data_yaml(dataset_root: Path, names: dict):
    lines = [
        "# YOLOv8 dataset configuration",
        f"path: {dataset_root.resolve().as_posix()}",
        "",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for k in sorted(names):
        lines.append(f"  {k}: {names[k]}")
    (dataset_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_structure(dataset_root: Path):
    names = parse_names_from_yaml(dataset_root / "data.yaml")

    for s in ["train", "val", "test"]:
        (dataset_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / s).mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": ["Train", "train"],
        "val": ["Validation", "Val", "val"],
        "test": ["Test", "test"],
    }

    for dst, srcs in split_map.items():
        for s in srcs:
            move_all_files(dataset_root / "images" / s, dataset_root / "images" / dst, IMG_EXTS)
            move_all_files(dataset_root / "labels" / s, dataset_root / "labels" / dst, {".txt"})

    cleanup_empty_dirs(dataset_root / "images")
    cleanup_empty_dirs(dataset_root / "labels")

    write_clean_data_yaml(dataset_root, names)


def main():
    parser = argparse.ArgumentParser(description="Unzip & normalize CVAT YOLO export (YOLOv8 clean).")
    parser.add_argument("--force", action="store_true", help="Hapus folder output jika sudah ada.")
    args = parser.parse_args()

    # ✅ Project root: .../triasetyani.thesis/
    project_root = Path(__file__).resolve().parents[2]  # CodePreprocessing -> Src -> ROOT
    datasets_dir = project_root / "Data" / "Datasets"

    zip_path = datasets_dir / "RBC_WBC_YOLO_DATASET.zip"
    out_dir = datasets_dir / "RBC_WBC_YOLO_DATASET"

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip tidak ditemukan: {zip_path}")

    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    dataset_root = infer_dataset_root(out_dir)

    # kalau data.yaml ada di subfolder
    if not (dataset_root / "data.yaml").exists():
        found = list(dataset_root.rglob("data.yaml"))
        if found:
            dataset_root = found[0].parent

    normalize_structure(dataset_root)

    print("✅ Dataset YOLOv8 CLEAN siap digunakan")
    print("📁 Dataset root :", dataset_root.resolve())
    print("📄 data.yaml    :", (dataset_root / "data.yaml").resolve())


if __name__ == "__main__":
    main()
