"""Evaluate chat luong search bang precision@K theo loai (species).

Chien luoc: lay 1 frame ngau nhien tu DB lam query, kiem tra top-K co bao nhieu
frame cung loai. Lap N lan, tinh precision@K trung binh.

Khong can ground truth - dung species cua frame query lam nhan.

Chay:
  python -m scripts.eval_search                  # 50 query, top_k=10
  python -m scripts.eval_search --n 100 --k 5    # 100 query, top_k=5
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.db import get_conn
from src.search import DEFAULT_WEIGHTS, SearchIndex


def fetch_random_query_frames(n: int) -> list[dict]:
    """Lay n frame ngau nhien tu DB lam query."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fr.id, fr.video_id, v.name, v.species, fr.timestamp_sec,
                   v.cloudinary_url
            FROM frames fr
            JOIN videos v ON v.id = fr.video_id
            ORDER BY random() LIMIT %s;
            """,
            (n,),
        )
        return [
            {"frame_id": str(r[0]), "video_id": str(r[1]), "video_name": r[2],
             "species": r[3], "timestamp_sec": float(r[4]), "url": r[5]}
            for r in cur.fetchall()
        ]


def extract_frame_at(url: str, ts: float) -> np.ndarray | None:
    """Download video tam thoi, seek den ts, doc 1 frame."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="So query")
    ap.add_argument("--k", type=int, default=10, help="Top-K cho precision@K")
    args = ap.parse_args()

    print("Loading search index...")
    idx = SearchIndex().load()

    print(f"\nFetching {args.n} random query frames...")
    queries = fetch_random_query_frames(args.n)

    p_at_k_list = []
    n_skipped = 0
    print(f"\nRunning {args.n} queries (top_k={args.k}):")
    print(f"{'#':>3}  {'species':20s}  {'video':30s}  p@k   correct/k")

    for i, q in enumerate(queries, 1):
        frame = extract_frame_at(q["url"], q["timestamp_sec"])
        if frame is None:
            n_skipped += 1
            print(f"{i:3d}  SKIP - khong load duoc frame")
            continue

        results = idx.search(frame, top_k=args.k + 1, weights=DEFAULT_WEIGHTS)
        # Bo result self-match (cung frame_id voi query)
        results = [r for r in results if r.frame_id != q["frame_id"]][:args.k]
        n_correct = sum(1 for r in results if r.species == q["species"])
        p_at_k = n_correct / args.k
        p_at_k_list.append(p_at_k)

        print(f"{i:3d}  {q['species'][:20]:20s}  {q['video_name'][:30]:30s}  "
              f"{p_at_k:.2f}  {n_correct}/{args.k}")

    if p_at_k_list:
        mean_p = sum(p_at_k_list) / len(p_at_k_list)
        print(f"\nMean precision@{args.k}: {mean_p:.3f}  "
              f"(over {len(p_at_k_list)} queries, {n_skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
