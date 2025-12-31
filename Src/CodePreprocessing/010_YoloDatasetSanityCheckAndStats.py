#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# data.yaml parser (minimal)
# -----------------------------
def parse_names_from_yaml(yaml_path: Path) -> Dict[int, str]:
    """
    Parse minimal bagian:
      names:
        0: RBC
        1: WBC
    Fallback jika tidak ada/invalid.
    """
    default = {0: "RBC", 1: "WBC"}
    if not yaml_path.exists():
        return default

    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    names: Dict[int, str] = {}
    in_names = False

    for line in lines:
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            # stop kalau ketemu top-level key lain
            if line and not line.startswith((" ", "\t")):
                break
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.strip().strip("\"'")  # bersihin quote
            if k.isdigit():
                names[int(k)] = v

    return names if names else default


# -----------------------------
# stats structures
# -----------------------------
@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    image_without_label: int = 0
    label_without_image: int = 0
    empty_label_files: int = 0
    invalid_label_lines: int = 0
    out_of_range_boxes: int = 0
    class_counts: Dict[int, int] = None

    def __post_init__(self):
        if self.class_counts is None:
            self.class_counts = {}


def stems_in_dir(dir_path: Path, exts: set[str]) -> Dict[str, Path]:
    """
    Mapping stem -> path, hanya untuk file dengan ekstensi tertentu.
    Non-recursive (sesuai struktur YOLOv8 standar).
    """
    out: Dict[str, Path] = {}
    if not dir_path.exists():
        return out
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def parse_yolo_line(line: str) -> Tuple[int, float, float, float, float] | None:
    """
    YOLO format: class x y w h
    """
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        return cls, x, y, w, h
    except ValueError:
        return None


def box_in_range(x: float, y: float, w: float, h: float) -> bool:
    """
    Check:
    - x,y,w,h berada di [0,1]
    - edges tidak keluar gambar (x±w/2, y±h/2)
    """
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        return False
    if (x - w / 2) < 0.0 or (x + w / 2) > 1.0:
        return False
    if (y - h / 2) < 0.0 or (y + h / 2) > 1.0:
        return False
    return True


def compute_split_stats(dataset_root: Path, split: str) -> Tuple[SplitStats, List[str], List[str]]:
    """
    Return:
      - SplitStats
      - list stems image tanpa label
      - list stems label tanpa image
    """
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    imgs = stems_in_dir(img_dir, IMG_EXTS)
    lbls = stems_in_dir(lbl_dir, {".txt"})

    st = SplitStats(images=len(imgs), labels=len(lbls))

    img_stems = set(imgs.keys())
    lbl_stems = set(lbls.keys())

    missing_lbl = sorted(img_stems - lbl_stems)
    missing_img = sorted(lbl_stems - img_stems)

    st.image_without_label = len(missing_lbl)
    st.label_without_image = len(missing_img)

    # parse each label file
    for stem, lbl_path in lbls.items():
        txt = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt == "":
            st.empty_label_files += 1
            continue

        for line in txt.splitlines():
            parsed = parse_yolo_line(line)
            if parsed is None:
                st.invalid_label_lines += 1
                continue

            cls, x, y, w, h = parsed
            st.class_counts[cls] = st.class_counts.get(cls, 0) + 1

            if not box_in_range(x, y, w, h):
                st.out_of_range_boxes += 1

    return st, missing_lbl, missing_img


def print_report(dataset_root: Path, names: Dict[int, str], sample_n: int = 10) -> int:
    splits = ["train", "val", "test"]
    all_stats: Dict[str, SplitStats] = {}

    print("=== YOLOv8 Dataset Sanity Check & Stats ===")
    print(f"Dataset root : {dataset_root.resolve()}")
    print(f"data.yaml     : {(dataset_root / 'data.yaml').resolve() if (dataset_root / 'data.yaml').exists() else '(not found)'}")
    print("")

    # per split
    for sp in splits:
        st, missing_lbl, missing_img = compute_split_stats(dataset_root, sp)
        all_stats[sp] = st

        print(f"[{sp.upper()}]")
        print(f"- images              : {st.images}")
        print(f"- labels              : {st.labels}")
        print(f"- image tanpa label   : {st.image_without_label}")
        print(f"- label tanpa image   : {st.label_without_image}")
        print(f"- empty label files   : {st.empty_label_files}")
        print(f"- invalid label lines : {st.invalid_label_lines}")
        print(f"- bbox out of range   : {st.out_of_range_boxes}")

        if st.class_counts:
            print("- class counts:")
            for cid in sorted(st.class_counts):
                cname = names.get(cid, f"class_{cid}")
                print(f"  - {cid} ({cname}): {st.class_counts[cid]}")
        else:
            print("- class counts: (none)")

        if missing_lbl:
            print(f"- sample image tanpa label (max{sample_n}): {missing_lbl[:sample_n]}")
        if missing_img:
            print(f"- sample label tanpa image (max{sample_n}): {missing_img[:sample_n]}")
        print("")

    # aggregate totals
    total_class: Dict[int, int] = {}
    for st in all_stats.values():
        for cid, c in st.class_counts.items():
            total_class[cid] = total_class.get(cid, 0) + c

    total_img_wo_lbl = sum(s.image_without_label for s in all_stats.values())
    total_lbl_wo_img = sum(s.label_without_image for s in all_stats.values())
    total_invalid = sum(s.invalid_label_lines for s in all_stats.values())
    total_oor = sum(s.out_of_range_boxes for s in all_stats.values())

    print("=== TOTAL (ALL SPLITS) ===")
    if total_class:
        for cid in sorted(total_class):
            cname = names.get(cid, f"class_{cid}")
            print(f"- {cid} ({cname}): {total_class[cid]}")
    else:
        print("- (no labels found)")
    print("")
    print("=== SUMMARY ===")
    print(f"- total image tanpa label : {total_img_wo_lbl}")
    print(f"- total label tanpa image : {total_lbl_wo_img}")
    print(f"- total invalid lines     : {total_invalid}")
    print(f"- total bbox out of range : {total_oor}")
    print("")

    # return code (opsional): 0 kalau aman, 1 kalau ada issue
    has_issue = (total_img_wo_lbl + total_lbl_wo_img + total_invalid + total_oor) > 0
    return 1 if has_issue else 0


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 dataset sanity check + class stats (terminal only).")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path root dataset YOLO (yang berisi images/ labels/ data.yaml). Default: <project_root>/Data/Datasets/RBC_WBC_YOLO_DATASET",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Jumlah sample stems yang ditampilkan untuk missing pairs (default=10).",
    )
    args = parser.parse_args()

    # Default mengikuti struktur project Anda: .../triasetyani.thesis/Src/... script ini
    project_root = Path(__file__).resolve().parents[2]
    default_dataset = project_root / "Data" / "Datasets" / "RBC_WBC_YOLO_DATASET"

    dataset_root = Path(args.dataset).expanduser() if args.dataset else default_dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root tidak ditemukan: {dataset_root}")

    names = parse_names_from_yaml(dataset_root / "data.yaml")
    rc = print_report(dataset_root, names, sample_n=args.sample)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
