"""Temporal features: Frame Difference (motion_magnitude, motion_area).

Chi dung khi build DB (so sanh frame voi frame truoc do).
KHONG dung trong similarity (query la anh tinh).
"""
from __future__ import annotations

import cv2
import numpy as np

MOTION_THRESHOLD = 20    # pixel diff > 20/255 thi tinh la "co thay doi"


def motion_features(
    frame_curr: np.ndarray, frame_prev: np.ndarray | None
) -> tuple[float | None, float | None]:
    """Tra ve (motion_magnitude, motion_area).

    None ca hai neu frame_prev=None (frame dau tien cua video).
    """
    if frame_prev is None:
        return None, None

    g_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(g_curr - g_prev)

    motion_magnitude = float(diff.mean() / 255.0)
    motion_area = float((diff > MOTION_THRESHOLD).mean())
    return motion_magnitude, motion_area
