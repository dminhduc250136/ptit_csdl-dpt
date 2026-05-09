"""Import data/extracted/*.json vao DB (frames + frame_features).

Idempotent: Skip video da co processed_at IS NOT NULL.
Chay 1 lan sau khi tat ca worker batch_extract da xong.

Chay: python -m scripts.import_extracted
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

from psycopg2.extras import execute_values

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.db import get_conn

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "extracted"


def load_processed_ids() -> set[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id::text FROM videos WHERE processed_at IS NOT NULL;")
        return {row[0] for row in cur.fetchall()}


def import_one(payload: dict) -> int:
    """Import 1 video JSON. Tra ve so frame da insert."""
    video_id = payload["video_id"]
    feature_version = payload["feature_version"]
    extraction_method = payload["extraction_method"]
    frames = payload["frames"]
    if not frames:
        return 0

    frame_rows = [
        (
            video_id,
            f["frame_index"],
            f["timestamp_sec"],
            extraction_method,
            None,                       # position_percent
            f["motion_magnitude"],
            f["motion_area"],
            None,                       # image_url
        )
        for f in frames
    ]

    with get_conn() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO frames (
                video_id, frame_index, timestamp_sec,
                extraction_method, position_percent,
                motion_magnitude, motion_area, image_url
            ) VALUES %s
            RETURNING id;""",
            frame_rows,
            page_size=200,
        )
        frame_ids = [r[0] for r in cur.fetchall()]

        feat_rows = [
            (
                fid, feature_version,
                f["color_histogram"], f["dominant_colors"],
                f["lbp_feature"], f["glcm_feature"],
                f["hog_feature"], f["edge_density"],
            )
            for fid, f in zip(frame_ids, frames)
        ]
        execute_values(
            cur,
            """INSERT INTO frame_features (
                frame_id, feature_version,
                color_histogram, dominant_colors,
                lbp_feature, glcm_feature,
                hog_feature, edge_density
            ) VALUES %s;""",
            feat_rows,
            page_size=100,
        )

        cur.execute(
            "UPDATE videos SET processed_at=now(), keyframe_count=%s WHERE id=%s;",
            (len(frames), video_id),
        )
    return len(frames)


def main() -> int:
    if not SRC_DIR.exists():
        print(f"Khong co thu muc {SRC_DIR}")
        return 1

    files = sorted(SRC_DIR.glob("*.json"))
    print(f"Found {len(files)} JSON files in {SRC_DIR}")

    processed = load_processed_ids()
    print(f"Da co {len(processed)} video processed -> skip\n")

    n_ok = n_skip = n_fail = total_kf = 0
    t_total = time.perf_counter()

    for i, path in enumerate(files, 1):
        video_id = path.stem
        prefix = f"[{i}/{len(files)}] {video_id[:8]}"

        if video_id in processed:
            n_skip += 1
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            n_frames = import_one(payload)
            n_ok += 1
            total_kf += n_frames
            if n_ok % 50 == 0:
                print(f"{prefix} OK ({payload['video_name'][:30]}) frames={n_frames} | "
                      f"running ok={n_ok} skip={n_skip}")
        except Exception as e:
            n_fail += 1
            print(f"{prefix} FAIL {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - t_total
    print(
        f"\nDone: {n_ok} OK, {n_skip} skipped, {n_fail} FAIL, "
        f"{total_kf} keyframes in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
