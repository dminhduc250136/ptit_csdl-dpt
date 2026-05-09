"""Color features: HSV histogram (96d) + K-means dominant colors (20d).

Ly do dung HSV thay RGB: tach mau (H) khoi do sang (V) -> on dinh hon
voi thay doi anh sang ngoai troi (sang/chieu/toi).
"""
from __future__ import annotations

import cv2
import numpy as np

HSV_BINS = 32                  # 32 bin moi kenh -> 96d total
KMEANS_K = 5                   # 5 mau chu dao -> 5 * (3 + 1) = 20d


def hsv_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """96d. Concat histogram H/S/V, normalize sum=1."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=HSV_BINS, range=(0, 180))
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=HSV_BINS, range=(0, 256))
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=HSV_BINS, range=(0, 256))
    feature = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return feature / (feature.sum() + 1e-7)


def dominant_colors(image_bgr: np.ndarray, k: int = KMEANS_K) -> np.ndarray:
    """20d. K-means k=5 -> [B,G,R, ratio] x 5, SORT theo ratio giam dan.

    Sort de output deterministic (kmeans random init co the doi thu tu cluster).
    """
    pixels = image_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    ratios = counts / counts.sum()

    # Sort cluster theo ratio giam dan
    order = np.argsort(-ratios)
    centers = centers[order]
    ratios = ratios[order]

    feature = np.empty(k * 4, dtype=np.float32)
    for i in range(k):
        feature[i * 4 : i * 4 + 3] = centers[i] / 255.0
        feature[i * 4 + 3] = ratios[i]
    return feature
