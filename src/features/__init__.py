"""Feature extractors cho CBVR.

Public API:
    extract_all(image_bgr) -> dict[str, np.ndarray]
        Tra ve dict 6 vector rieng (color_histogram, dominant_colors,
        lbp_feature, glcm_feature, hog_feature, edge_density).
"""
from .extractor import FEATURE_DIMS, extract_all

__all__ = ["extract_all", "FEATURE_DIMS"]
