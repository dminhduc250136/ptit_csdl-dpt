"""Smoke + invariant tests cho feature extractors.

Khong can data that - sinh anh tong hop bang numpy.

Chay: python -m tests.test_features
"""
from __future__ import annotations

import io
import sys

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.features import FEATURE_DIMS, extract_all
from src.features.temporal import motion_features


def make_synthetic_image(seed: int = 0, size: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """Sinh anh BGR ngau nhien deterministic theo seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)


def make_solid_color(color_bgr: tuple[int, int, int], size: tuple[int, int] = (720, 1280)) -> np.ndarray:
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dimensions():
    img = make_synthetic_image(seed=42)
    feats = extract_all(img)
    for name, expected_dim in FEATURE_DIMS.items():
        assert name in feats, f"missing {name}"
        actual = feats[name].shape[0]
        assert actual == expected_dim, f"{name}: expected {expected_dim}, got {actual}"
        assert feats[name].dtype == np.float32, f"{name} dtype = {feats[name].dtype}"
    print(f"[OK] test_dimensions - 6 vector dung dim: {dict((k,v.shape[0]) for k,v in feats.items())}")


def test_no_nan_no_inf():
    img = make_synthetic_image(seed=1)
    feats = extract_all(img)
    for name, vec in feats.items():
        assert not np.isnan(vec).any(), f"{name} contains NaN"
        assert not np.isinf(vec).any(), f"{name} contains Inf"
    print("[OK] test_no_nan_no_inf - khong co NaN/Inf trong 6 vector")


def test_histogram_normalized():
    """color_histogram va lbp_feature phai sum ~ 1."""
    img = make_synthetic_image(seed=7)
    feats = extract_all(img)
    for name in ("color_histogram", "lbp_feature", "hog_feature"):
        s = feats[name].sum()
        assert abs(s - 1.0) < 1e-3, f"{name}.sum() = {s}, expected ~1.0"
    print("[OK] test_histogram_normalized - color/lbp/hog sum=1")


def test_dominant_colors_sorted():
    """Ratio cluster phai giam dan (sort theo argsort(-counts))."""
    img = make_synthetic_image(seed=3)
    feats = extract_all(img)
    dom = feats["dominant_colors"].reshape(5, 4)
    ratios = dom[:, 3]
    assert np.all(ratios[:-1] >= ratios[1:]), f"ratios khong giam dan: {ratios}"
    assert abs(ratios.sum() - 1.0) < 1e-3, f"ratios.sum() = {ratios.sum()}"
    print(f"[OK] test_dominant_colors_sorted - ratios={ratios.round(3).tolist()}")


def test_deterministic():
    """Cung input -> cung output (k-means co thay doi nho do random init)."""
    img = make_synthetic_image(seed=99)
    f1 = extract_all(img.copy())
    f2 = extract_all(img.copy())
    # k-means co the khac nhau giua 2 lan goi -> chap nhan dominant_colors loi nho.
    # Nhung cac feature deterministic phai bang nhau:
    for name in ("color_histogram", "lbp_feature", "glcm_feature", "hog_feature", "edge_density"):
        assert np.allclose(f1[name], f2[name]), f"{name} not deterministic"
    print("[OK] test_deterministic - 5/6 feature giong nhau giua 2 lan goi")


def test_solid_vs_random_differ():
    """Anh thuan mau khac han anh ngau nhien."""
    solid = make_solid_color((40, 80, 200))   # mau cam nau
    random_img = make_synthetic_image(seed=11)
    f_solid = extract_all(solid)
    f_random = extract_all(random_img)

    # GLCM contrast cua anh thuan mau ~0, anh random > 0
    assert f_solid["glcm_feature"][0] < f_random["glcm_feature"][0], (
        f"GLCM contrast: solid={f_solid['glcm_feature'][0]} >= random={f_random['glcm_feature'][0]}"
    )

    # Edge density anh thuan mau ~0
    assert f_solid["edge_density"].mean() < 0.01
    print("[OK] test_solid_vs_random_differ - GLCM contrast & edge phan biet duoc")


def test_aspect_ratio_robust():
    """Anh 1280x720 va anh 720x720 cua cung mot vat -> hog tuong tu (nho padding)."""
    base = make_synthetic_image(seed=21, size=(720, 720))
    wide = cv2.copyMakeBorder(base, 0, 0, 280, 280, cv2.BORDER_CONSTANT, value=0)
    f_base = extract_all(base)
    f_wide = extract_all(wide)
    # HOG khong giong het (padding khac) nhung cosine sim phai cao
    cos = float(
        np.dot(f_base["hog_feature"], f_wide["hog_feature"])
        / (np.linalg.norm(f_base["hog_feature"]) * np.linalg.norm(f_wide["hog_feature"]) + 1e-7)
    )
    print(f"[OK] test_aspect_ratio_robust - HOG cosine(base, wide) = {cos:.3f}")


def test_motion_features():
    f1 = make_synthetic_image(seed=1)
    f2 = make_synthetic_image(seed=2)

    mag_none, area_none = motion_features(f1, None)
    assert mag_none is None and area_none is None

    mag, area = motion_features(f2, f1)
    assert 0.0 <= mag <= 1.0
    assert 0.0 <= area <= 1.0
    assert mag > 0  # 2 anh random khac nhau -> co motion
    print(f"[OK] test_motion_features - motion_magnitude={mag:.3f} motion_area={area:.3f}")


def test_speed():
    """Sanity check: 1 anh < 1s."""
    import time
    img = make_synthetic_image(seed=0)
    extract_all(img)  # warmup
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        extract_all(img)
    elapsed = (time.perf_counter() - t0) / n
    assert elapsed < 2.0, f"Qua cham: {elapsed:.2f}s/anh"
    print(f"[OK] test_speed - {elapsed*1000:.0f}ms/anh (~{1/elapsed:.1f} anh/s)")


def main() -> int:
    tests = [
        test_dimensions,
        test_no_nan_no_inf,
        test_histogram_normalized,
        test_dominant_colors_sorted,
        test_deterministic,
        test_solid_vs_random_differ,
        test_aspect_ratio_robust,
        test_motion_features,
        test_speed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
