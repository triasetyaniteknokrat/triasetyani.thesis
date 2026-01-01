#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tkinter Microscope GUI - Realtime Preview + REC Toggle (Thesis-ready)
- Buttons: Load Model, Open Media, Open Camera, Reset Position, Stop, REC
- Camera mode:
  - Open Camera => preview + detection (no recording)
  - REC ON => start session recording (CSV+MP4) + watermark counts in recorded video
  - REC OFF => stop recording (camera preview continues)
  - Stop => stop worker completely
- Watermark (recorded video):
  Total objects, FPS_STREAM (EMA), FPS_INFER (1/dt), inference ms, per-class counts
- Output root:
  Data/DataTesting/Output/Realtime/<session_folder>/
- Session folder:
  <YYYYMMDD_HHMMSS>_cam<idx>_<modelstem>
"""

from __future__ import annotations

import re
import time
import csv
import queue
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # pip install pillow


# ---------------- CONFIG ----------------
SQUARE = 720
GAP = 5
WINDOW_WIDTH = SQUARE * 2 + GAP + 40

DEFAULT_CAMERA_W = 1280
DEFAULT_CAMERA_H = 720
DEFAULT_CAMERA_FPS = 30  # target fps (device may ignore)

DEFAULT_CONF = 0.5
DEFAULT_IMGSZ = 640

BUFFER_FRAMES_FOR_FPS = 30
MIN_WRITER_FPS = 5.0
MAX_WRITER_FPS = 60.0

BTN_W = 18   # Tk units (text-based)
BTN_H = 2

# smoothing untuk FPS_STREAM (EMA)
FPS_EMA_ALPHA = 0.2


# ---------------- Helpers ----------------
def center_crop_square(img_bgr: np.ndarray, size=SQUARE) -> np.ndarray:
    if img_bgr is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    if w == h == size:
        return img_bgr.copy()

    cx, cy = w // 2, h // 2
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = x1 + size, y1 + size
    if x2 > w:
        x2, x1 = w, max(0, w - size)
    if y2 > h:
        y2, y1 = h, max(0, h - size)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def label_color(name: str):
    h = abs(hash(name))
    return (int(h % 200) + 30, int((h // 200) % 200) + 30, int((h // 40000) % 200) + 30)


def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s.strip("_")


def ts_ms() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ---------------- ViewTransform (zoom/pan) ----------------
@dataclass
class ViewState:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


def view_to_rect(view: ViewState, canvas_w: int, canvas_h: int, img_w: int, img_h: int):
    x1 = int(round((0 - view.offset_x) / view.scale))
    y1 = int(round((0 - view.offset_y) / view.scale))
    x2 = int(round((canvas_w - view.offset_x) / view.scale))
    y2 = int(round((canvas_h - view.offset_y) / view.scale))

    x1 = clamp(x1, 0, img_w)
    y1 = clamp(y1, 0, img_h)
    x2 = clamp(x2, 0, img_w)
    y2 = clamp(y2, 0, img_h)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, img_w, img_h
    return (x1, y1, x2, y2)


# ---------------- ImageCanvas (zoom/pan) ----------------
class ImageCanvas(tk.Canvas):
    def __init__(self, master, width=SQUARE, height=SQUARE, **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=1, highlightbackground="#444", **kwargs)
        self.configure(bg="black")

        self.view = ViewState()
        self._img_rgb = None
        self._photo = None
        self._img_id = None

        self._dragging = False
        self._last_x = 0
        self._last_y = 0

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        # Windows/Mac wheel
        self.bind("<MouseWheel>", self._on_wheel)
        # Linux wheel
        self.bind("<Button-4>", self._on_wheel_linux_up)
        self.bind("<Button-5>", self._on_wheel_linux_down)

    def set_image_rgb(self, img_rgb: np.ndarray, reset=False):
        self._img_rgb = img_rgb
        if reset:
            self.view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._redraw()

    def set_view(self, scale: float, offset_x: float, offset_y: float):
        self.view.scale = float(scale)
        self.view.offset_x = float(offset_x)
        self.view.offset_y = float(offset_y)
        self._redraw()

    def reset_view(self):
        self.view = ViewState(scale=1.0, offset_x=0.0, offset_y=0.0)
        self._redraw()

    def get_visible_rect(self):
        if self._img_rgb is None:
            return None
        h, w = self._img_rgb.shape[:2]
        cw = int(self.cget("width"))
        ch = int(self.cget("height"))
        return view_to_rect(self.view, cw, ch, w, h)

    def _redraw(self):
        if self._img_rgb is None:
            self.delete("all")
            self._img_id = None
            self._photo = None
            return

        img = self._img_rgb
        h, w = img.shape[:2]
        scaled_w = max(1, int(round(w * self.view.scale)))
        scaled_h = max(1, int(round(h * self.view.scale)))

        pil = Image.fromarray(img)
        pil_scaled = pil.resize((scaled_w, scaled_h), resample=Image.BILINEAR)

        cw = int(self.cget("width"))
        ch = int(self.cget("height"))
        bg = Image.new("RGB", (cw, ch), (0, 0, 0))

        ox = int(round(self.view.offset_x))
        oy = int(round(self.view.offset_y))
        bg.paste(pil_scaled, (ox, oy))

        self._photo = ImageTk.PhotoImage(bg)
        if self._img_id is None:
            self._img_id = self.create_image(0, 0, image=self._photo, anchor="nw")
        else:
            self.itemconfig(self._img_id, image=self._photo)

    def _on_press(self, e):
        if self._img_rgb is None:
            return
        self._dragging = True
        self._last_x, self._last_y = e.x, e.y

    def _on_drag(self, e):
        if not self._dragging or self._img_rgb is None:
            return
        dx = e.x - self._last_x
        dy = e.y - self._last_y
        self.view.offset_x += dx
        self.view.offset_y += dy
        self._last_x, self._last_y = e.x, e.y
        self._redraw()
        self.event_generate("<<ViewChanged>>", when="tail")

    def _on_release(self, _e):
        self._dragging = False

    def _apply_zoom(self, factor: float, mx: float, my: float):
        if self._img_rgb is None:
            return
        old_scale = self.view.scale
        new_scale = clamp(old_scale * factor, 0.2, 6.0)
        if abs(new_scale - old_scale) < 1e-9:
            return

        old_img_x = (mx - self.view.offset_x) / old_scale
        old_img_y = (my - self.view.offset_y) / old_scale
        self.view.scale = new_scale
        self.view.offset_x = mx - old_img_x * new_scale
        self.view.offset_y = my - old_img_y * new_scale

        self._redraw()
        self.event_generate("<<ViewChanged>>", when="tail")

    def _on_wheel(self, e):
        factor = 1.15 if e.delta > 0 else 0.85
        self._apply_zoom(factor, e.x, e.y)

    def _on_wheel_linux_up(self, e):
        self._apply_zoom(1.15, e.x, e.y)

    def _on_wheel_linux_down(self, e):
        self._apply_zoom(0.85, e.x, e.y)


# ---------------- Worker Thread ----------------
class InferenceWorker(threading.Thread):
    def __init__(self, model: YOLO, device: str, project_root: Path, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.project_root = project_root
        self.out_queue = out_queue

        self._running = threading.Event()
        self._running.set()

        self._mode = None
        self._input_path = None
        self._cam_index = None
        self._crop_rect = None

        # recording state
        self._recording = False
        self._recording_lock = threading.Lock()
        self._recording_request = None  # None/True/False

        self.session_dir = None
        self.csv_path = None
        self.video_path = None
        self._csv_file = None
        self._csv_writer = None
        self._writer = None

        self._buffer_frames = []
        self._buffer_start_time = None

        try:
            self.class_names = list(self.model.names.values())
        except Exception:
            self.class_names = []

        self._frame_idx = 0

        # for image mode re-infer
        self._trigger = threading.Event()
        self._cached_image_sq = None

        # FPS stream (wall-clock) tracking
        self._prev_frame_t = None
        self._fps_stream_ema = 0.0

    # ----- config -----
    def configure_camera(self, index: int):
        self._mode = "camera"
        self._cam_index = index

    def configure_video(self, path: str):
        self._mode = "video"
        self._input_path = path

    def configure_image(self, path: str):
        self._mode = "image"
        self._input_path = path

    def set_crop_rect(self, rect):
        self._crop_rect = rect

    def trigger_inference(self):
        self._trigger.set()

    def stop(self):
        self._running.clear()
        self._trigger.set()
        self.request_recording(False)

    # ----- recording control -----
    def request_recording(self, enable: bool):
        with self._recording_lock:
            self._recording_request = bool(enable)

    # ----- internals -----
    def _infer(self, bgr_img):
        t0 = time.time()
        results = self.model(
            bgr_img,
            device=self.device,
            verbose=False,
            conf=DEFAULT_CONF,
            imgsz=DEFAULT_IMGSZ
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        return results, dt

    def _draw_boxes(self, base_bgr, results):
        ann = base_bgr.copy()
        res = results[0].to("cpu")
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ann

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = res.names

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            label = names.get(int(cls), str(cls))
            col = label_color(label)
            cv2.rectangle(ann, (x1i, y1i), (x2i, y2i), col, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            by = max(0, y1i - th - 6)
            cv2.rectangle(ann, (x1i, by), (x1i + tw + 6, by + th + 4), col, -1)
            cv2.putText(ann, text, (x1i + 3, by + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return ann

    def _counts_from_results(self, results) -> dict:
        res = results[0].to("cpu")
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {}
        labels = boxes.cls.cpu().numpy().astype(int)
        names = res.names
        counts = {}
        for l in labels:
            n = names.get(int(l), str(l))
            counts[n] = counts.get(n, 0) + 1
        return counts

    def _update_fps_stream(self):
        """Update FPS stream EMA using wall-clock time between processed frames."""
        now = time.time()
        if self._prev_frame_t is None:
            self._prev_frame_t = now
            return 0.0

        dt_stream = max(1e-6, now - self._prev_frame_t)
        fps_stream = 1.0 / dt_stream

        # EMA smoothing
        self._fps_stream_ema = (FPS_EMA_ALPHA * fps_stream) + ((1.0 - FPS_EMA_ALPHA) * self._fps_stream_ema)
        self._prev_frame_t = now
        return self._fps_stream_ema

    def _add_watermark_counts(self, frame_bgr: np.ndarray, counts: dict, dt_infer: float, fps_stream: float) -> np.ndarray:
        if frame_bgr is None:
            return frame_bgr

        overlay = frame_bgr.copy()
        h, w = overlay.shape[:2]

        total = 0
        lines = []
        for cn in self.class_names:
            v = int(counts.get(cn, 0))
            total += v
            lines.append(f"{cn}: {v}")

        fps_infer = (1.0 / dt_infer) if dt_infer > 0 else 0.0
        inf_ms = dt_infer * 1000.0

        header = f"Total: {total} | FPS_STREAM: {fps_stream:.1f} | FPS_INFER: {fps_infer:.1f} | INF: {inf_ms:.1f} ms"
        lines = [header] + lines

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        pad = 10
        line_gap = 6

        text_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
        box_w = max(ts[0] for ts in text_sizes) + pad * 2
        box_h = sum(ts[1] for ts in text_sizes) + pad * 2 + line_gap * (len(lines) - 1)

        x0, y0 = 10, 10
        x1, y1 = x0 + box_w, y0 + box_h
        x1 = min(x1, w - 1)
        y1 = min(y1, h - 1)

        alpha = 0.45
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

        y = y0 + pad + text_sizes[0][1]
        for i, t in enumerate(lines):
            cv2.putText(out, t, (x0 + pad, y), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
            if i < len(lines) - 1:
                y += text_sizes[i + 1][1] + line_gap

        return out

    def _make_session_dir(self) -> Path:
        out_root = (self.project_root / "Data" / "DataTesting" / "Output" / "Realtime").resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        ts_folder = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_stem = "model"
        try:
            p = getattr(self.model, "ckpt_path", None)
            if p:
                model_stem = Path(p).stem
        except Exception:
            pass

        session_folder = f"{ts_folder}_cam{self._cam_index}_{safe_slug(model_stem)}"
        session_dir = (out_root / session_folder).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _start_recording(self):
        if self._recording:
            return

        self.session_dir = self._make_session_dir()
        self.csv_path = self.session_dir / "session.csv"
        self.video_path = self.session_dir / "session.mp4"

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        header = ["frame_idx", "timestamp", "inference_ms", "fps_infer", "fps_stream"]
        for cn in self.class_names:
            header.append(f"count_{cn}")
        header.append("total_objects")
        self._csv_writer.writerow(header)

        self._buffer_frames = []
        self._buffer_start_time = time.time()
        self._writer = None

        self._frame_idx = 0
        self._recording = True

        self.out_queue.put(("session_started", str(self.session_dir)))

    def _init_writer_if_ready(self):
        if self._writer is not None:
            return
        if len(self._buffer_frames) < BUFFER_FRAMES_FOR_FPS:
            return

        elapsed = max(1e-6, time.time() - (self._buffer_start_time or time.time()))
        fps_est = len(self._buffer_frames) / elapsed
        fps_est = float(np.clip(fps_est, MIN_WRITER_FPS, MAX_WRITER_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.video_path), fourcc, fps_est, (SQUARE, SQUARE))

        for fr in self._buffer_frames:
            self._writer.write(fr)
        self._buffer_frames.clear()

    def _stop_recording(self):
        if not self._recording:
            return

        try:
            if self._writer is None:
                elapsed = max(1e-6, time.time() - (self._buffer_start_time or time.time()))
                fps_fallback = (len(self._buffer_frames) / elapsed) if self._buffer_frames else 10.0
                fps_fallback = float(np.clip(fps_fallback, MIN_WRITER_FPS, MAX_WRITER_FPS))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.video_path), fourcc, fps_fallback, (SQUARE, SQUARE))
                for fr in self._buffer_frames:
                    self._writer.write(fr)
                self._buffer_frames.clear()
        except Exception:
            pass

        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
        self._writer = None

        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None

        finished_dir = str(self.session_dir) if self.session_dir else ""
        self.session_dir = None
        self.csv_path = None
        self.video_path = None

        self._recording = False
        self.out_queue.put(("session_stopped", finished_dir))

    def _apply_recording_request_if_any(self):
        with self._recording_lock:
            req = self._recording_request
            self._recording_request = None

        if req is None:
            return
        if req is True and self._mode == "camera":
            self._start_recording()
        elif req is False:
            self._stop_recording()

    def _process_one_frame(self, frame_sq_bgr: np.ndarray):
        # update fps stream once per processed frame (wall-clock)
        fps_stream = self._update_fps_stream()

        orig_rgb = cv2.cvtColor(frame_sq_bgr, cv2.COLOR_BGR2RGB)

        rect = self._crop_rect
        if rect:
            x1, y1, x2, y2 = rect
            crop = frame_sq_bgr[y1:y2, x1:x2].copy()
            img_for_infer = crop if crop.size else frame_sq_bgr
            offset_x, offset_y = x1, y1
        else:
            img_for_infer = frame_sq_bgr
            offset_x, offset_y = 0, 0

        results, dt_infer = self._infer(img_for_infer)
        ann_bgr = self._draw_boxes(img_for_infer, results)

        if offset_x != 0 or offset_y != 0:
            full_annot_rgb = cv2.cvtColor(frame_sq_bgr, cv2.COLOR_BGR2RGB)
            ah, aw = ann_bgr.shape[:2]
            full_annot_rgb[offset_y:offset_y+ah, offset_x:offset_x+aw] = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

            ann_for_video_bgr = frame_sq_bgr.copy()
            ann_for_video_bgr[offset_y:offset_y+ah, offset_x:offset_x+aw] = ann_bgr
        else:
            full_annot_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            ann_for_video_bgr = ann_bgr

        counts = self._counts_from_results(results)

        # preview frames -> GUI
        self.out_queue.put(("frame", orig_rgb, full_annot_rgb, counts, dt_infer, self._mode))

        # recording
        if self._recording and self._csv_writer is not None:
            self._frame_idx += 1
            inference_ms = dt_infer * 1000.0
            fps_infer = (1.0 / dt_infer) if dt_infer > 0 else 0.0

            row = [self._frame_idx, ts_ms(), round(inference_ms, 3), round(fps_infer, 3), round(fps_stream, 3)]
            total_objects = 0
            for cn in self.class_names:
                v = int(counts.get(cn, 0))
                row.append(v)
                total_objects += v
            row.append(total_objects)
            self._csv_writer.writerow(row)

            ann_for_video_bgr_wm = self._add_watermark_counts(ann_for_video_bgr, counts, dt_infer, fps_stream)

            if self._writer is None:
                self._buffer_frames.append(ann_for_video_bgr_wm)
                self._init_writer_if_ready()
            else:
                self._writer.write(ann_for_video_bgr_wm)

    def run(self):
        if self.model is None:
            self.out_queue.put(("error", "Model belum dimuat."))
            return

        try:
            if self._mode == "camera":
                cap = cv2.VideoCapture(self._cam_index if self._cam_index is not None else 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_H)
                cap.set(cv2.CAP_PROP_FPS, DEFAULT_CAMERA_FPS)  # may be ignored by device/driver

                if not cap.isOpened():
                    self.out_queue.put(("error", "Gagal membuka kamera."))
                    return

                # reset fps tracking when camera starts
                self._prev_frame_t = None
                self._fps_stream_ema = 0.0

                while self._running.is_set():
                    self._apply_recording_request_if_any()

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_sq = center_crop_square(frame, SQUARE)
                    self._process_one_frame(frame_sq)
                    time.sleep(0.001)

                cap.release()
                if self._recording:
                    self._stop_recording()

            elif self._mode == "video":
                cap = cv2.VideoCapture(str(self._input_path))
                if not cap.isOpened():
                    self.out_queue.put(("error", "Gagal membuka video."))
                    return

                self._prev_frame_t = None
                self._fps_stream_ema = 0.0

                while self._running.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_sq = center_crop_square(frame, SQUARE)
                    self._process_one_frame(frame_sq)
                    time.sleep(0.001)

                cap.release()

            elif self._mode == "image":
                img = cv2.imread(str(self._input_path))
                if img is None:
                    self.out_queue.put(("error", "Gagal membaca gambar."))
                    return
                frame_sq = center_crop_square(img, SQUARE)
                self._cached_image_sq = frame_sq

                # for image, fps_stream isn't meaningful, but we still keep it stable
                self._prev_frame_t = None
                self._fps_stream_ema = 0.0

                self._process_one_frame(frame_sq)

                while self._running.is_set():
                    self._trigger.wait(timeout=0.5)
                    if not self._running.is_set():
                        break
                    if self._trigger.is_set():
                        self._trigger.clear()
                        if self._cached_image_sq is not None:
                            self._process_one_frame(self._cached_image_sq)

        except Exception as e:
            self.out_queue.put(("error", str(e)))
            try:
                if self._recording:
                    self._stop_recording()
            except Exception:
                pass


# ---------------- Camera selection dialog ----------------
class CameraSelectDialog(tk.Toplevel):
    def __init__(self, master, cameras: list[int]):
        super().__init__(master)
        self.title("Pilih Kamera")
        self.configure(bg="#f8f8f8")
        self.resizable(False, False)
        self.selected = None

        tk.Label(self, text="Kamera terdeteksi:", bg="#f8f8f8", fg="#222",
                 font=("Arial", 11)).pack(padx=12, pady=(12, 6))

        self.listbox = tk.Listbox(self, width=30, height=min(8, max(3, len(cameras))), font=("Arial", 11))
        for i in cameras:
            self.listbox.insert("end", f"Camera {i}")
        self.listbox.pack(padx=12, pady=6)
        self.listbox.selection_set(0)

        btns = tk.Frame(self, bg="#f8f8f8")
        btns.pack(pady=(6, 12))

        tk.Button(btns, text="OK", width=10, command=self._ok).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", width=10, command=self._cancel).pack(side="left", padx=6)

        self.listbox.bind("<Double-Button-1>", lambda _e: self._ok())

        self.update_idletasks()
        px = master.winfo_rootx()
        py = master.winfo_rooty()
        mw = master.winfo_width()
        mh = master.winfo_height()
        w = self.winfo_width()
        h = self.winfo_height()
        self.geometry(f"+{px + (mw - w)//2}+{py + (mh - h)//2}")

        self.transient(master)
        self.grab_set()

    def _ok(self):
        sel = self.listbox.curselection()
        if not sel:
            self.selected = None
        else:
            text = self.listbox.get(sel[0])
            self.selected = int(text.split()[-1])
        self.destroy()

    def _cancel(self):
        self.selected = None
        self.destroy()


# ---------------- Main Window ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("DORIS JUARSA - 24321025")
        self.geometry(f"{WINDOW_WIDTH}x{SQUARE+250}")
        self.configure(bg="#f8f8f8")
        self.resizable(False, False)

        # Root project path
        self.project_root = Path(__file__).resolve().parents[2]
        self.default_dialog_dir = self.project_root  # always reset dialogs to this

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: YOLO | None = None
        self.worker: InferenceWorker | None = None

        self.out_q = queue.Queue()

        # ---------- Banner ----------
        banner = tk.Frame(self, bg="#f8f8f8")
        banner.pack(pady=(10, 6), padx=10, fill="x")

        banner_lbl = tk.Label(
            banner,
            text=(
                "Model Deteksi Objek Real-Time Untuk Klasifikasi Otomatis Subtipe Leukosit "
                "Pada Citra Mikroskopis Darah Dengan YOLOv8\n"
                "Dibuat oleh: Doris Juarsa | NPM: 24321025\n"
                "Pembimbing: Dr. Erliyan Redy Susanto, S.Kom., M.Kom. | Penguji 1: Dr. Rohmat Indra Borman, M.Kom."
            ),
            bg="#f8f8f8",
            fg="#222",
            justify="center",
            font=("Arial", 11)
        )
        banner_lbl.pack()

        # ---------- Buttons ----------
        btn_frame = tk.Frame(self, bg="#f8f8f8")
        btn_frame.pack(pady=6)

        self.btn_load = tk.Button(btn_frame, text="Load Model", width=BTN_W, height=BTN_H, command=self.load_model)
        self.btn_media = tk.Button(btn_frame, text="Open Media", width=BTN_W, height=BTN_H, command=self.open_media, state="disabled")
        self.btn_cam = tk.Button(btn_frame, text="Open Camera", width=BTN_W, height=BTN_H, command=self.open_camera, state="disabled")
        self.btn_reset = tk.Button(btn_frame, text="Reset Position", width=BTN_W, height=BTN_H, command=self.reset_views, state="disabled")
        self.btn_stop = tk.Button(btn_frame, text="Stop", width=BTN_W, height=BTN_H, command=self.stop_worker, state="disabled")

        self.rec_on = False
        self.btn_rec = tk.Button(btn_frame, text="REC: OFF", width=BTN_W, height=BTN_H, command=self.toggle_rec, state="disabled")

        for w in [self.btn_load, self.btn_media, self.btn_cam, self.btn_reset, self.btn_stop, self.btn_rec]:
            w.pack(side="left", padx=4)

        # ---------- Canvases ----------
        canvas_frame = tk.Frame(self, bg="#f8f8f8")
        canvas_frame.pack(pady=(8, 4), padx=10)

        self.left = ImageCanvas(canvas_frame, width=SQUARE, height=SQUARE)
        self.right = ImageCanvas(canvas_frame, width=SQUARE, height=SQUARE)

        self.left.pack(side="left")
        tk.Frame(canvas_frame, width=GAP, height=SQUARE, bg="#f8f8f8").pack(side="left")
        self.right.pack(side="left")

        self.left.bind("<<ViewChanged>>", self.on_view_changed)

        # ---------- Class counters ----------
        self.counter_frame = tk.Frame(self, bg="#f8f8f8")
        self.counter_frame.pack(pady=(4, 6))
        self.class_labels: dict[str, tk.Label] = {}

        # ---------- Status ----------
        self.status_var = tk.StringVar(value="")
        status = tk.Label(self, textvariable=self.status_var, bg="#f8f8f8", fg="#333", font=("Arial", 11))
        status.pack(pady=(4, 8))

        self.after(15, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI state ----------
    def _enable_after_model(self, enable=True):
        state = "normal" if enable else "disabled"
        for w in [self.btn_media, self.btn_cam, self.btn_reset, self.btn_stop, self.btn_rec]:
            w.configure(state=state)

    def _set_rec_ui(self, is_on: bool):
        self.rec_on = bool(is_on)
        self.btn_rec.configure(
            text=("REC: ON" if self.rec_on else "REC: OFF"),
            relief=("sunken" if self.rec_on else "raised"),
            bg=("#ffd0d0" if self.rec_on else self.cget("bg"))
        )

    # ---------- Model ----------
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Pilih Model YOLOv8 (.pt/.onnx)",
            initialdir=str(self.default_dialog_dir),
            filetypes=[("Model Files", "*.pt *.onnx")]
        )
        if not path:
            return
        try:
            self.model = YOLO(path)
            self._enable_after_model(True)

            # rebuild class counters
            for child in self.counter_frame.winfo_children():
                child.destroy()
            self.class_labels.clear()

            class_names = list(self.model.names.values())
            for name in class_names:
                lbl = tk.Label(self.counter_frame, text=f"{name}: 0", bg="#ffffff", fg="#222",
                               relief="solid", bd=1, padx=8, pady=4, font=("Arial", 10))
                lbl.pack(side="left", padx=4)
                self.class_labels[name] = lbl

            self.status_var.set(f"Model dimuat: {Path(path).name} ({self.device}) | conf={DEFAULT_CONF}")

        except Exception as e:
            messagebox.showerror("Gagal muat model", str(e))
            self.model = None
            self._enable_after_model(False)

    # ---------- Media ----------
    def open_media(self):
        if self.model is None:
            messagebox.showwarning("Peringatan", "Muat model terlebih dahulu.")
            return

        path = filedialog.askopenfilename(
            title="Pilih Media (Image/Video)",
            initialdir=str(self.default_dialog_dir),
            filetypes=[("Media", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return

        ext = Path(path).suffix.lower()
        self._stop_worker_internal()

        self.worker = InferenceWorker(
            model=self.model,
            device=self.device,
            project_root=self.project_root,
            out_queue=self.out_q
        )

        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            self.worker.configure_video(path)
            self.status_var.set("Video mode: detecting... (Stop untuk berhenti)")
        else:
            self.worker.configure_image(path)
            self.status_var.set("Image mode: pan/zoom untuk re-run detection.")

        self.worker.start()
        self.btn_stop.configure(state="normal")
        self._set_rec_ui(False)

    # ---------- Camera ----------
    def detect_available_cameras(self, max_tests=8):
        cams = []
        for i in range(max_tests):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cams.append(i)
            cap.release()
        return cams

    def open_camera(self):
        if self.model is None:
            messagebox.showwarning("Peringatan", "Muat model terlebih dahulu.")
            return

        cams = self.detect_available_cameras()
        if not cams:
            messagebox.showwarning("No Camera", "Tidak ada kamera terdeteksi.")
            return

        dlg = CameraSelectDialog(self, cams)
        self.wait_window(dlg)
        cam_index = dlg.selected
        if cam_index is None:
            return

        self._stop_worker_internal()

        self.worker = InferenceWorker(
            model=self.model,
            device=self.device,
            project_root=self.project_root,
            out_queue=self.out_q
        )
        self.worker.configure_camera(cam_index)
        self.worker.start()

        self.btn_stop.configure(state="normal")
        self._set_rec_ui(False)
        self.status_var.set(f"Camera {cam_index} preview + detect (tekan REC untuk mulai rekam)")

    # ---------- View sync & crop rect ----------
    def on_view_changed(self, _evt=None):
        rect = self.left.get_visible_rect()
        if self.worker and rect is not None:
            self.worker.set_crop_rect(rect)
            if getattr(self.worker, "_mode", None) == "image":
                self.worker.trigger_inference()

        self.right.set_view(self.left.view.scale, self.left.view.offset_x, self.left.view.offset_y)

    # ---------- REC ----------
    def toggle_rec(self):
        next_state = not self.rec_on

        if not self.worker or not self.worker.is_alive() or getattr(self.worker, "_mode", None) != "camera":
            self._set_rec_ui(False)
            messagebox.showwarning("REC", "REC hanya bisa digunakan saat mode Camera aktif.")
            return

        self.worker.request_recording(next_state)
        self._set_rec_ui(next_state)
        self.status_var.set("REC ● Recording started..." if next_state else "REC ■ Recording stopped (preview continues)")

    # ---------- Reset / Stop ----------
    def reset_views(self):
        self.left.reset_view()
        self.right.reset_view()
        self.on_view_changed()

    def stop_worker(self):
        self._stop_worker_internal()
        self.status_var.set("Stopped.")
        self._set_rec_ui(False)

    def _stop_worker_internal(self):
        if self.worker and self.worker.is_alive():
            try:
                self.worker.request_recording(False)
            except Exception:
                pass
            self.worker.stop()
            self.worker.join(timeout=2.5)
        self.worker = None
        self.btn_stop.configure(state="disabled")

    # ---------- Queue polling ----------
    def _poll_queue(self):
        try:
            while True:
                item = self.out_q.get_nowait()
                kind = item[0]

                if kind == "frame":
                    _, orig_rgb, ann_rgb, counts, dt_infer, mode = item
                    self.left.set_image_rgb(orig_rgb, reset=False)
                    self.right.set_image_rgb(ann_rgb, reset=False)
                    self._update_counters(counts)

                elif kind == "session_started":
                    _, session_dir = item
                    self._set_rec_ui(True)
                    self.status_var.set(f"REC ● Started: {session_dir}")

                elif kind == "session_stopped":
                    _, session_dir = item
                    self._set_rec_ui(False)
                    if session_dir:
                        messagebox.showinfo("Recording selesai", f"Output tersimpan di:\n{session_dir}")
                        self.status_var.set("REC ■ Recording stopped (preview continues)")
                    else:
                        self.status_var.set("REC ■ Recording stopped")

                elif kind == "error":
                    _, msg = item
                    self._set_rec_ui(False)
                    messagebox.showerror("Worker error", msg)
                    self.status_var.set("Error: " + msg)

        except queue.Empty:
            pass

        self.after(15, self._poll_queue)

    def _update_counters(self, counts: dict):
        for name, lbl in self.class_labels.items():
            val = int(counts.get(name, 0))
            lbl.configure(text=f"{name}: {val}")
            if val > 0:
                lbl.configure(bg="#d0ffd0", fg="#000")
            else:
                lbl.configure(bg="#ffffff", fg="#222")

    def on_close(self):
        self._stop_worker_internal()
        self.destroy()


def main():
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
