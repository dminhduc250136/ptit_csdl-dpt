"""Keyframe extractor: scene-aware delta histogram.

Lay frame khi:
  1. Histogram thay doi dang ke (scene change), HOAC
  2. Da du khoang thoi gian toi thieu (min_interval).

Co cap max_frames de tranh video dai sinh qua nhieu keyframe
(vd video 600s khong nen sinh 300 frame).

Tra ve list[Keyframe] - moi item co frame_index, timestamp_sec, frame_bgr.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Keyframe:
    frame_index: int
    timestamp_sec: float
    frame_bgr: np.ndarray


# Tham so theo docs/cbvr_context.md
THRESHOLD = 0.4         # cosine distance > threshold -> scene change
MIN_INTERVAL_SEC = 2.0  # khoang cach toi thieu giua 2 keyframe lien tiep
MAX_FRAMES = 15         # tranh video qua dai sinh qua nhieu keyframe
HIST_BINS = 32          # bin moi kenh HSV de tinh delta histogram


def _scene_histogram(frame_bgr: np.ndarray) -> np.ndarray:
    """HSV histogram nhe (32 bin per channel) cho phat hien scene change."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, _ = np.histogram(hsv[:, :, 0], bins=HIST_BINS, range=(0, 180))
    s, _ = np.histogram(hsv[:, :, 1], bins=HIST_BINS, range=(0, 256))
    v, _ = np.histogram(hsv[:, :, 2], bins=HIST_BINS, range=(0, 256))
    feature = np.concatenate([h, s, v]).astype(np.float32)
    return feature / (feature.sum() + 1e-7)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-9:
        return 0.0
    return 1.0 - dot / norm


def extract_keyframes(
    video_path: str,
    threshold: float = THRESHOLD,
    min_interval_sec: float = MIN_INTERVAL_SEC,
    max_frames: int = MAX_FRAMES,
) -> list[Keyframe]:
    """Tach keyframe theo scene-aware delta histogram.

    Logic:
      - Frame dau tien luon duoc lay (lam anchor).
      - Tu frame thu 2: lay neu scene_changed AND interval_ok.
        ('AND' - khong phai 'OR' - de tranh lay qua nhieu).
      - Cap max_frames: dung khi du.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Sample 1 frame moi ~0.5s (nua min_interval) - du de phat hien scene change
    # ma khong can decode het 30 frame/giay. Voi fps=30 -> step=15.
    sample_step = max(1, int(round(fps * min_interval_sec / 4.0)))

    keyframes: list[Keyframe] = []
    prev_hist: np.ndarray | None = None
    last_saved_time = -min_interval_sec
    frame_idx = 0

    try:
        while True:
            # Skip cv2.CAP_PROP_POS_FRAMES seek vi mot so codec khong support chinh xac.
            # Thay bang cap.grab() (decode header only) skip nhanh hon read() ~5x.
            for _ in range(sample_step - 1):
                if not cap.grab():
                    cap.release()
                    return keyframes
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            curr_hist = _scene_histogram(frame)

            should_save = False
            if prev_hist is None:
                should_save = True
            else:
                interval_ok = (current_time - last_saved_time) >= min_interval_sec
                if interval_ok:
                    diff = _cosine_distance(curr_hist, prev_hist)
                    scene_changed = diff > threshold
                    should_save = scene_changed or (
                        current_time - last_saved_time >= 2 * min_interval_sec
                    )

            if should_save:
                keyframes.append(
                    Keyframe(
                        frame_index=frame_idx,
                        timestamp_sec=float(current_time),
                        frame_bgr=frame.copy(),
                    )
                )
                last_saved_time = current_time
                if len(keyframes) >= max_frames:
                    break

            prev_hist = curr_hist
            frame_idx += 1
    finally:
        cap.release()

    return keyframes
