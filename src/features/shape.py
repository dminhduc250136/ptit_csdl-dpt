"""Shape features: HOG (576d, cell_size=8) + Edge Density 4x4 (16d).

QUAN TRONG:
- Resize co padding (khong stretch) -> tranh meo hinh khi aspect ratio le.
- cell_size=8 (baseline trong thiet_ke_he_thong_cbvr.md): 8x8 cell x 9 bin = 576d.
  Voi cosine-per-group + weighted sum thi dim khong gay mat can bang giua cac nhom,
  HOG dim cao chi anh huong toc do extract va storage.
"""
from __future__ import annotations

import cv2
import numpy as np

HOG_TARGET = (64, 64)
HOG_CELL = 8
HOG_BINS = 9
HOG_DIM = (HOG_TARGET[0] // HOG_CELL) * (HOG_TARGET[1] // HOG_CELL) * HOG_BINS  # 576

EDGE_GRID = 4
EDGE_DIM = EDGE_GRID * EDGE_GRID  # 16


def _resize_with_padding(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Resize giu nguyen aspect ratio, pad den size dich."""
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


def hog_feature(image_bgr: np.ndarray) -> np.ndarray:
    """144d. HOG voi cell_size=16, 9 bin orientation (0-180 do)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    padded = _resize_with_padding(gray, HOG_TARGET)

    gx = cv2.Sobel(padded, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(padded, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0
    bin_idx = np.minimum((angle / (180.0 / HOG_BINS)).astype(np.int32), HOG_BINS - 1)

    th, tw = HOG_TARGET
    cells_y, cells_x = th // HOG_CELL, tw // HOG_CELL
    feature = np.zeros((cells_y, cells_x, HOG_BINS), dtype=np.float32)

    for cy in range(cells_y):
        for cx in range(cells_x):
            y0, y1 = cy * HOG_CELL, (cy + 1) * HOG_CELL
            x0, x1 = cx * HOG_CELL, (cx + 1) * HOG_CELL
            cell_mag = magnitude[y0:y1, x0:x1]
            cell_bin = bin_idx[y0:y1, x0:x1]
            for b in range(HOG_BINS):
                feature[cy, cx, b] = cell_mag[cell_bin == b].sum()

    flat = feature.ravel()
    return flat / (flat.sum() + 1e-7)


def edge_density(image_bgr: np.ndarray) -> np.ndarray:
    """16d. Canny -> chia 4x4 -> mat do canh moi vung."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    cy, cx = h // EDGE_GRID, w // EDGE_GRID
    out = np.empty(EDGE_DIM, dtype=np.float32)
    for i in range(EDGE_GRID):
        for j in range(EDGE_GRID):
            cell = edges[i * cy : (i + 1) * cy, j * cx : (j + 1) * cx]
            out[i * EDGE_GRID + j] = cell.mean() / 255.0
    return out
