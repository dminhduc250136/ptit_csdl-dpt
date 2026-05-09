"""Grid search weights de tim combo tot nhat theo precision@K.

Load index 1 lan, lay N query frame (cache feature vector), chay nhieu combo
weight cung 1 query set -> compare apple-to-apple.

Chay: python -m scripts.tune_weights --n 80 --k 10
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.db import get_conn
from src.features import extract_all
from src.search import SearchIndex, _strip_flat_bin


def fetch_random_query_frames(n: int) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fr.id, fr.video_id, v.name, v.species, fr.timestamp_sec, v.cloudinary_url
            FROM frames fr JOIN videos v ON v.id=fr.video_id
            ORDER BY random() LIMIT %s;
            """,
            (n,),
        )
        return [
            {"frame_id": str(r[0]), "video_id": str(r[1]), "video_name": r[2],
             "species": r[3], "timestamp_sec": float(r[4]), "url": r[5]}
            for r in cur.fetchall()
        ]


def extract_query_frame(url: str, ts: float) -> np.ndarray | None:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    finally:
        path.unlink(missing_ok=True)


def evaluate(
    idx: SearchIndex,
    query_features: list[dict],
    k: int,
    weights: dict[str, float],
) -> float:
    """Tinh mean p@k cho 1 weight combo. query_features = list dict
    {frame_id, species, q_color, q_texture, q_shape (already L2 normalized)}."""
    p_list = []
    for q in query_features:
        sim = (
            weights["color"]   * (idx.color   @ q["q_color"])
          + weights["texture"] * (idx.texture @ q["q_texture"])
          + weights["shape"]   * (idx.shape   @ q["q_shape"])
        )
        # Top-K (k+1 vi can bo self-match)
        kk = min(k + 1, len(sim) - 1)
        top_idx = np.argpartition(-sim, kk)[:kk + 1]
        top_idx = top_idx[np.argsort(-sim[top_idx])]
        # Bo self
        top_idx = [i for i in top_idx if idx.frame_ids[i] != q["frame_id"]][:k]
        n_correct = sum(1 for i in top_idx if idx.species[i] == q["species"])
        p_list.append(n_correct / k)
    return float(np.mean(p_list))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="So query frame")
    ap.add_argument("--k", type=int, default=10, help="Top-K")
    args = ap.parse_args()

    print("Loading search index...")
    idx = SearchIndex().load()

    print(f"\nFetching {args.n} random query frames...")
    queries = fetch_random_query_frames(args.n)

    print(f"Extracting features for {len(queries)} queries (download + extract)...")
    t0 = time.perf_counter()
    qf: list[dict] = []
    for i, q in enumerate(queries, 1):
        frame = extract_query_frame(q["url"], q["timestamp_sec"])
        if frame is None:
            continue
        feats = extract_all(frame)
        feats["lbp_feature"] = _strip_flat_bin(feats["lbp_feature"])
        q_c = np.concatenate([feats["color_histogram"], feats["dominant_colors"]])
        q_t = np.concatenate([feats["lbp_feature"], feats["glcm_feature"]])
        q_s = np.concatenate([feats["hog_feature"], feats["edge_density"]])
        qf.append({
            "frame_id": q["frame_id"],
            "species": q["species"],
            "q_color": q_c / (np.linalg.norm(q_c) + 1e-9),
            "q_texture": q_t / (np.linalg.norm(q_t) + 1e-9),
            "q_shape": q_s / (np.linalg.norm(q_s) + 1e-9),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(queries)} done")
    print(f"Done {len(qf)} queries in {time.perf_counter()-t0:.1f}s")

    # ---------------------------------------------------------------
    # 1. Per-group baseline: each group alone
    # ---------------------------------------------------------------
    print(f"\n=== Per-group baseline (chi 1 nhom co weight=1) ===")
    print(f"{'config':40s}  p@{args.k}")
    print("-" * 55)
    for w in [
        {"color": 1.0, "texture": 0.0, "shape": 0.0},
        {"color": 0.0, "texture": 1.0, "shape": 0.0},
        {"color": 0.0, "texture": 0.0, "shape": 1.0},
    ]:
        p = evaluate(idx, qf, args.k, w)
        label = "color only" if w["color"] else ("texture only" if w["texture"] else "shape only")
        print(f"{label:40s}  {p:.3f}")

    # Default
    default = {"color": 0.35, "texture": 0.35, "shape": 0.20}
    p_default = evaluate(idx, qf, args.k, default)
    print(f"{'DEFAULT (35/35/20)':40s}  {p_default:.3f}")

    # ---------------------------------------------------------------
    # 2. Grid search
    # ---------------------------------------------------------------
    print(f"\n=== Grid search ===")
    grid = []
    # Step 0.1, sum = 1
    for c10 in range(0, 11):
        for t10 in range(0, 11 - c10):
            s10 = 10 - c10 - t10
            grid.append((c10 / 10, t10 / 10, s10 / 10))

    results = []
    print(f"Evaluating {len(grid)} weight combos on {len(qf)} queries...")
    for c, t, s in grid:
        if c == 0 and t == 0 and s == 0:
            continue
        p = evaluate(idx, qf, args.k, {"color": c, "texture": t, "shape": s})
        results.append((p, c, t, s))

    # Top 10
    results.sort(reverse=True)
    print(f"\nTop 10 weight combos by p@{args.k}:")
    print(f"{'rank':>4}  {'color':>5}  {'texture':>7}  {'shape':>5}  p@{args.k}")
    print("-" * 50)
    for i, (p, c, t, s) in enumerate(results[:10], 1):
        marker = "  <- DEFAULT" if (abs(c - 0.35) < 0.01 and abs(t - 0.35) < 0.01 and abs(s - 0.20) < 0.01) else ""
        print(f"{i:>4}  {c:>5.2f}  {t:>7.2f}  {s:>5.2f}  {p:.3f}{marker}")

    print(f"\nWorst 5:")
    for p, c, t, s in results[-5:]:
        print(f"      {c:>5.2f}  {t:>7.2f}  {s:>5.2f}  {p:.3f}")

    print(f"\nBest:    {results[0][0]:.3f}  (color={results[0][1]}, texture={results[0][2]}, shape={results[0][3]})")
    print(f"Default: {p_default:.3f}  (color=0.35, texture=0.35, shape=0.20)")
    print(f"Lift:    +{(results[0][0] - p_default)*100:.1f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
