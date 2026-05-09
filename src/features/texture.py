"""Texture features: Uniform LBP (59d) + GLCM (4d).

Su dung skimage.feature thay loop Python -> nhanh ~100x.
Uniform LBP (P=8, R=1) on dinh hon voi noise anh sang.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

LBP_P = 8                # so neighbor
LBP_R = 1                # ban kinh
LBP_BINS = LBP_P + 2     # 10 voi P=8? Khong - method='uniform' cho P+2=10... thuc te...
                         # voi 'uniform': so bin = P*(P-1) + 3 = 59 voi P=8
LBP_N_UNIFORM = LBP_P * (LBP_P - 1) + 3   # = 59 voi P=8

GLCM_LEVELS = 8


def lbp_feature(image_bgr: np.ndarray) -> np.ndarray:
    """59d. Uniform LBP histogram, normalize sum=1.

    Voi P=8, method='uniform': output co 59 bin (58 uniform pattern + 1 non-uniform).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(), bins=LBP_N_UNIFORM, range=(0, LBP_N_UNIFORM)
    )
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-7)


def glcm_feature(image_bgr: np.ndarray) -> np.ndarray:
    """4d. GLCM contrast, energy, homogeneity, correlation."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Quantize ve 8 muc de giam kich thuoc ma tran (bat buoc cho graycomatrix)
    gray_q = (gray // (256 // GLCM_LEVELS)).astype(np.uint8)

    # distance=1, angle=0 (huong ngang). symmetric=True -> doi xung.
    glcm = graycomatrix(
        gray_q,
        distances=[1],
        angles=[0],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    return np.array(
        [contrast, energy, homogeneity, correlation], dtype=np.float32
    )
