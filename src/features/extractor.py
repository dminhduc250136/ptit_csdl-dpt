"""Orchestrator: extract tat ca 6 vector rieng cho 1 anh.

Tra ve dict de luu vao 6 cot trong frame_features.
KHONG normalize min-max o day - normalize se fit tren toan bo DB sau khi extract xong
(theo khuyen nghi 'fit scaler tren toan bo DB' trong docs).
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import dominant_colors, hsv_histogram
from .shape import edge_density, hog_feature
from .texture import glcm_feature, lbp_feature

INPUT_RESIZE = (256, 256)   # resize input -> on dinh ket qua va nhanh hon

FEATURE_DIMS = {
    "color_histogram": 96,
    "dominant_colors": 20,
    "lbp_feature": 59,
    "glcm_feature": 4,
    "hog_feature": 576,
    "edge_density": 16,
}


def _resize_with_padding(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    th, tw = target
    h, w = image.shape[:2]
    scale = min(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_h, pad_w = th - nh, tw - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def extract_all(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Extract 6 vector tu 1 anh BGR.

    Resize input ve 256x256 voi padding de:
      1. Tieu chuan hoa kich thuoc (anh query va frame video cung scale).
      2. Tang toc (anh 1280x720 -> 256x256 ~25x nhanh hon cho LBP/GLCM).
    HOG co rieng resize 64x64 ben trong, khong bi anh huong.
    """
    img = _resize_with_padding(image_bgr, INPUT_RESIZE)

    return {
        "color_histogram": hsv_histogram(img),
        "dominant_colors": dominant_colors(img),
        "lbp_feature": lbp_feature(img),
        "glcm_feature": glcm_feature(img),
        "hog_feature": hog_feature(img),
        "edge_density": edge_density(img),
    }
