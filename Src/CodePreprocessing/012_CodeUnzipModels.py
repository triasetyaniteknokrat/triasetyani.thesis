#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
012_CodeUnzipModels.py

- Mengekstrak file ZIP di folder Data/DataModels/runs/
- Membuat folder tujuan dengan nama yang sama dengan file ZIP
- Menghindari struktur folder bersarang (nested)
"""

# Import library standard untuk ekstraksi
import zipfile
from pathlib import Path
import sys

# ==================================================
# TARGET PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataModels"
    / "runs"
).resolve()

# List file yang akan diekstrak
ZIP_FILES = [
    "RBC_WBC_YOLO_DATASET_640_yolov8s_mosaic.zip",
    "RBC_WBC_YOLO_DATASET_640_yolov8s_no_mosaic.zip"
]

# ==================================================
# UNZIP PROCESS
# ==================================================
print(f"📂 Mencari file di: {TARGET_DIR}")

for zip_name in ZIP_FILES:
    zip_path = TARGET_DIR / zip_name
    
    # Folder tujuan (nama file tanpa .zip)
    # Misal: .../runs/nama_file/
    extract_folder = TARGET_DIR / zip_path.stem

    print("-" * 50)
    
    if not zip_path.exists():
        print(f"❌ File tidak ditemukan: {zip_name}")
        continue

    if extract_folder.exists():
        print(f"⚠️ Folder tujuan sudah ada: {extract_folder.name}")
        print("跳過 ekstraksi untuk menghindari penumpukan data.")
        continue

    print(f"📦 Mengekstrak: {zip_name}")
    print(f"➡️ Ke folder : {extract_folder.name}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Ekstrak langsung ke folder yang baru dibuat
            zip_ref.extractall(extract_folder)
        print(f"✅ Berhasil diekstrak.")
    except Exception as e:
        print(f"❌ Gagal mengekstrak {zip_name}: {e}")

print("-" * 50)
print("🚀 Semua proses unzip selesai.")