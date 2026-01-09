#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
011_CodeDownloadModels.py

- Mengunduh dua dataset YOLO (mosaic & no-mosaic)
- Lokasi Output: Data/DataModels/runs/
- Menggunakan wget dengan proteksi overwrite
"""

from pathlib import Path
import subprocess
import sys

# ==================================================
# BASE PATH
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# DATASET & MODELS PATH
# ==================================================
# Target: triasetyani.thesis/Data/DataModels/runs/
TARGET_DIR = (
    BASE_DIR
    / ".."
    / ".."
    / "Data"
    / "DataModels"
    / "runs"
).resolve()

# List URL yang akan didownload
DOWNLOAD_LINKS = [
    "https://serverdorisjuarsafoldershare.dorisjuarsa.com/TriaSetyani/RBC_WBC_YOLO_DATASET_640_yolov8s_mosaic.zip",
    "https://serverdorisjuarsafoldershare.dorisjuarsa.com/TriaSetyani/RBC_WBC_YOLO_DATASET_640_yolov8s_no_mosaic.zip"
]

# ==================================================
# PREPARATION
# ==================================================
# Buat folder jika belum ada
TARGET_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Folder target: {TARGET_DIR}")

# ==================================================
# DOWNLOAD PROCESS
# ==================================================
for url in DOWNLOAD_LINKS:
    # Ambil nama file dari URL
    filename = url.split("/")[-1]
    dest_path = TARGET_DIR / filename
    
    print("-" * 50)
    if dest_path.exists():
        print(f"⚠️ File sudah ada: {filename}")
        print("跳過 download untuk file ini.")
        continue

    print(f"⬇️ Sedang mengunduh: {filename}")
    try:
        subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                url,
                "-O",
                str(dest_path)
            ],
            check=True
        )
        print(f"✅ Berhasil mengunduh: {filename}")
    except subprocess.CalledProcessError:
        print(f"❌ Gagal mengunduh: {filename}")
    except Exception as e:
        print(f"❌ Error tidak terduga: {e}")

print("-" * 50)
print("🚀 Proses selesai.")