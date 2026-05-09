"""Batch extract: download -> keyframe -> feature -> ghi file JSON.

KHONG insert DB. Chi ghi `data/extracted/{video_id}.json` cho moi video xong.
Sau khi tat ca worker xong, chay scripts.import_extracted de import vao DB.

Sharding cho parallel: --shard 1/4 -> chi xu ly video co
hash(uuid) % 4 == 0. Mo 4 terminal voi 1/4, 2/4, 3/4, 4/4.

File ton tai = video da xu ly -> resume tu nhien.

Chay:
  python -m scripts.batch_extract                       # 5 video (smoke test)
  python -m scripts.batch_extract --all                 # tat ca, 1 worker
  python -m scripts.batch_extract --all --shard 1/4     # chi shard 1
  python -m scripts.batch_extract --all --shard 2/4     # chi shard 2
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.db import get_conn
from src.features import extract_all
from src.features.temporal import motion_features
from src.keyframe import extract_keyframes

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "extracted"

DOWNLOAD_TIMEOUT = 60
DOWNLOAD_RETRIES = 3
TARGET_W, TARGET_H = 1280, 720
FEATURE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Sharding helper
# ---------------------------------------------------------------------------

def parse_shard(s: str | None) -> tuple[int, int] | None:
    """'1/4' -> (0, 4). Tra ve None neu khong co."""
    if not s:
        return None
    a, b = s.split("/")
    idx, total = int(a), int(b)
    if not (1 <= idx <= total):
        raise ValueError(f"--shard {s} khong hop le")
    return idx - 1, total


def in_shard(uuid_str: str, shard: tuple[int, int] | None) -> bool:
    if shard is None:
        return True
    idx, total = shard
    h = int(hashlib.md5(uuid_str.encode()).hexdigest(), 16)
    return h % total == idx


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def download(url: str, dst: Path) -> None:
    last_err = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
                r.raise_for_status()
                with dst.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download failed sau {DOWNLOAD_RETRIES} lan: {last_err}")


def maybe_resize_video(src: Path, needs_resize: bool) -> Path:
    if not needs_resize:
        return src

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = src.with_suffix(".resized.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (TARGET_W, TARGET_H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = min(TARGET_W / w, TARGET_H / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        pad_w, pad_h = TARGET_W - nw, TARGET_H - nh
        padded = cv2.copyMakeBorder(
            resized, pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=0,
        )
        writer.write(padded)
    cap.release()
    writer.release()
    return out_path


def fetch_pending(shard: tuple[int, int] | None, max_videos: int | None) -> list[dict]:
    """Lay video chua co file output trong shard nay."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in OUT_DIR.glob("*.json")}

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, species, cloudinary_url, needs_resize, duration_sec
            FROM videos
            WHERE processed_at IS NULL
            ORDER BY created_at;
            """
        )
        all_pending = cur.fetchall()

    out: list[dict] = []
    for row in all_pending:
        vid = str(row[0])
        if vid in done:
            continue
        if not in_shard(vid, shard):
            continue
        out.append({
            "id": vid, "name": row[1], "species": row[2],
            "url": row[3], "needs_resize": row[4], "duration": row[5],
        })
        if max_videos is not None and len(out) >= max_videos:
            break
    return out


def process_video(video: dict) -> dict:
    """Process 1 video, ghi ket qua ra file JSON. Tra ve dict thong ke."""
    stats = {"name": video["name"], "n_keyframes": 0, "ok": False, "error": None}
    out_file = OUT_DIR / f"{video['id']}.json"

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        raw = tmp_dir / "video.mp4"
        t0 = time.perf_counter()
        download(video["url"], raw)
        stats["t_download"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proc = maybe_resize_video(raw, video["needs_resize"])
        stats["t_resize"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        keyframes = extract_keyframes(str(proc))
        stats["t_keyframe"] = time.perf_counter() - t0
        stats["n_keyframes"] = len(keyframes)

        if not keyframes:
            stats["error"] = "no_keyframes"
            return stats

        t0 = time.perf_counter()
        records: list[dict] = []
        prev_bgr: np.ndarray | None = None
        for kf in keyframes:
            feats = extract_all(kf.frame_bgr)
            mag, area = motion_features(kf.frame_bgr, prev_bgr)
            records.append({
                "frame_index": int(kf.frame_index),
                "timestamp_sec": float(kf.timestamp_sec),
                "motion_magnitude": mag,
                "motion_area": area,
                "color_histogram": feats["color_histogram"].tolist(),
                "dominant_colors": feats["dominant_colors"].tolist(),
                "lbp_feature": feats["lbp_feature"].tolist(),
                "glcm_feature": feats["glcm_feature"].tolist(),
                "hog_feature": feats["hog_feature"].tolist(),
                "edge_density": feats["edge_density"].tolist(),
            })
            prev_bgr = kf.frame_bgr
        stats["t_feature"] = time.perf_counter() - t0

    # Ghi atomic: tmp file -> rename. Tranh worker khac doc file dang ghi do.
    payload = {
        "video_id": video["id"],
        "video_name": video["name"],
        "species": video["species"],
        "feature_version": FEATURE_VERSION,
        "extraction_method": "scene_aware",
        "extracted_at": time.time(),
        "frames": records,
    }
    tmp_out = out_file.with_suffix(".json.tmp")
    with tmp_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    tmp_out.replace(out_file)

    stats["ok"] = True
    return stats


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Chay tat ca video con lai")
    ap.add_argument("--limit", type=int, default=5, help="So video chay (default 5)")
    ap.add_argument("--shard", type=str, default=None,
                    help="vd '1/4' = chi xu ly video shard 1 trong 4 worker")
    ap.add_argument("--worker", type=str, default=None, help="Ten worker (chi de log)")
    args = ap.parse_args()

    worker = args.worker or f"pid{os.getpid()}"
    shard = parse_shard(args.shard)
    max_videos = None if args.all else args.limit

    videos = fetch_pending(shard, max_videos)
    shard_str = f" shard={shard[0]+1}/{shard[1]}" if shard else ""
    print(f"[{worker}{shard_str}] Pending: {len(videos)} videos\n")

    n_ok = n_fail = total_kf = 0
    t_total = time.perf_counter()
    for i, v in enumerate(videos, 1):
        prefix = f"[{worker} {i}/{len(videos)}] {v['name'][:38]:38s} ({v['species']})"
        try:
            s = process_video(v)
            if s["ok"]:
                n_ok += 1
                total_kf += s["n_keyframes"]
                print(
                    f"{prefix} OK  kf={s['n_keyframes']:2d}  "
                    f"dl={s['t_download']:4.1f}s  kf={s['t_keyframe']:4.1f}s  "
                    f"feat={s['t_feature']:4.1f}s"
                )
            else:
                n_fail += 1
                print(f"{prefix} SKIP {s.get('error')}")
        except Exception as e:
            n_fail += 1
            print(f"{prefix} FAIL {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - t_total
    print(
        f"\n[{worker}] Done: {n_ok} OK, {n_fail} FAIL, {total_kf} keyframes "
        f"in {elapsed:.0f}s ({elapsed/max(n_ok,1):.1f}s/video)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
