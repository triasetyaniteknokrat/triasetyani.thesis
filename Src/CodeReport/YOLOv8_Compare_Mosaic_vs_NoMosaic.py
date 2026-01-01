# 007_YOLOv8_Compare_Mosaic_vs_NoMosaic.py

from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# ==================================================
# BASE PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# DATASETS ROOT
# ==================================================
DATASETS_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "Datasets"
).resolve()

# ==================================================
# DATA MODELS ROOT
# ==================================================
MODELS_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataModels"
).resolve()

# ==================================================
# OUTPUT DIR
# ==================================================
OUTPUT_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataTesting"
    / "Output"
    / "YOLOv8_Comparison"
    / "Mosaic_vs_NoMosaic"
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# EDIT THESE (sesuaikan dengan project Anda)
# ==================================================
# Nama folder run di Data/DataModels/runs/...
RUN_NO_MOSAIC = "RBC_WBC_YOLO_DATASET_640_yolov8s_no_mosaic"     
RUN_MOSAIC    = "RBC_WBC_YOLO_DATASET_640_yolov8s_mosaic"    

# Nama folder dataset di Data/Datasets/...
DATASET_NAME = "RBC_WBC_YOLO_DATASET"          

# ==================================================
# MODEL PATHS
# ==================================================
MODEL_NO_MOSAIC = (
    MODELS_DIR
    / "runs"
    / RUN_NO_MOSAIC
    / "weights"
    / "best.pt"
)

MODEL_MOSAIC = (
    MODELS_DIR
    / "runs"
    / RUN_MOSAIC
    / "weights"
    / "best.pt"
)

# ==================================================
# DATASET PATHS
# ==================================================
DATA_ROOT = DATASETS_DIR / DATASET_NAME
DATA_YAML = DATA_ROOT / "data.yaml"

# ==================================================
# SCENARIO CONFIG
# (untuk tesis: model A vs model B pada dataset yang sama, split test)
# ==================================================
SCENARIOS = [
    ("NoMosaic_on_Test", MODEL_NO_MOSAIC, DATA_YAML),
    ("Mosaic_on_Test",   MODEL_MOSAIC,    DATA_YAML),
]

# ==================================================
# Helpers
# ==================================================
def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"[NOT FOUND] {label}: {path}")

def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)

# ==================================================
# Pre-check paths
# ==================================================
assert_exists(DATA_ROOT, "DATA_ROOT")
assert_exists(DATA_YAML, "DATA_YAML")
assert_exists(MODEL_NO_MOSAIC, "MODEL_NO_MOSAIC(best.pt)")
assert_exists(MODEL_MOSAIC, "MODEL_MOSAIC(best.pt)")

# ==================================================
# EVALUATION
# ==================================================
results_summary = []
results_per_class = []  # optional (jika tersedia)

IMG_SIZE = 640
CONF_TH = 0.25
IOU_TH = 0.5

for name, model_path, data_yaml in SCENARIOS:
    print(f"\n🚀 Evaluating: {name}")
    print(f"   Model: {model_path}")
    print(f"   Data : {data_yaml}")

    model = YOLO(model_path)

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMG_SIZE,
        conf=CONF_TH,
        iou=IOU_TH,
        plots=True,
        project=str(OUTPUT_DIR),
        name=name
    )

    # ---- overall metrics (macro)
    precision = float(metrics.box.mp)     # mean precision
    recall    = float(metrics.box.mr)     # mean recall
    f1        = 2 * safe_div(precision * recall, precision + recall)
    map50     = float(metrics.box.map50)
    map5095   = float(metrics.box.map)

    results_summary.append({
        "Scenario": name,
        "Model": model_path.parent.parent.name,  # nama folder run
        "Dataset": DATA_ROOT.name,
        "Split": "test",
        "imgsz": IMG_SIZE,
        "conf": CONF_TH,
        "iou": IOU_TH,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1-score": round(f1, 6),
        "mAP@0.5": round(map50, 6),
        "mAP@0.5:0.95": round(map5095, 6),
    })

    # ---- per-class metrics (kalau Ultralytics version Anda menyediakan)
    # Beberapa versi ada: metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap
    # Jika tidak ada, bagian ini dilewati tanpa error.
    names = getattr(metrics, "names", None)  # dict id->name
    try:
        p_cls    = getattr(metrics.box, "p", None)
        r_cls    = getattr(metrics.box, "r", None)
        ap50_cls = getattr(metrics.box, "ap50", None)
        ap_cls   = getattr(metrics.box, "ap", None)

        if names is not None and p_cls is not None and r_cls is not None:
            # names biasanya dict {0:"WBC",1:"RBC",...}
            for cid, cname in names.items():
                cid = int(cid)
                pc = float(p_cls[cid]) if ap50_cls is not None else float(p_cls[cid])
                rc = float(r_cls[cid])
                f1c = 2 * safe_div(pc * rc, pc + rc)

                row = {
                    "Scenario": name,
                    "ClassID": cid,
                    "ClassName": cname,
                    "Precision": round(pc, 6),
                    "Recall": round(rc, 6),
                    "F1-score": round(f1c, 6),
                }

                if ap50_cls is not None:
                    row["AP@0.5"] = round(float(ap50_cls[cid]), 6)
                if ap_cls is not None:
                    row["AP@0.5:0.95"] = round(float(ap_cls[cid]), 6)

                results_per_class.append(row)

    except Exception as e:
        print(f"   (info) Per-class metrics not available or failed to parse: {e}")

# ==================================================
# SAVE SUMMARY
# ==================================================
df_sum = pd.DataFrame(results_summary)
csv_path = OUTPUT_DIR / "YOLOv8_Comparison_Mosaic_vs_NoMosaic.csv"
xlsx_path = OUTPUT_DIR / "YOLOv8_Comparison_Mosaic_vs_NoMosaic.xlsx"
df_sum.to_csv(csv_path, index=False)

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df_sum.to_excel(writer, index=False, sheet_name="Summary")
    if len(results_per_class) > 0:
        df_cls = pd.DataFrame(results_per_class)
        df_cls.to_excel(writer, index=False, sheet_name="PerClass")

print("\n✅ Evaluation completed!")
print(f"📄 Summary CSV : {csv_path}")
print(f"📊 Summary XLSX: {xlsx_path}")
print(f"📁 Plots saved under: {OUTPUT_DIR}")
