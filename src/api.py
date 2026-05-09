"""FastAPI /search endpoint.

POST /search
  multipart/form-data:
    image:       file (jpg/png) - bat buoc
    top_k:       int            - default 10
    color:       float          - weight color, default 0.0
    texture:     float          - weight texture, default 0.9
    shape:       float          - weight shape, default 0.1
    by_video:    bool           - true = group ket qua theo video, default true

Response: list ket qua + thoi gian xu ly + log_id.

Index load 1 lan khi server start (lifespan), tat ca request dung chung.
"""
from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.db import get_conn
from src.search import DEFAULT_WEIGHTS, SearchIndex, group_by_video


# ---------------------------------------------------------------------------
# Lifespan: load index 1 lan khi server start
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading search index...")
    t0 = time.perf_counter()
    app.state.index = SearchIndex().load()
    print(f"[startup] Index ready in {time.perf_counter() - t0:.2f}s")
    yield


app = FastAPI(title="CBVR Search API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # cho phep test tu localhost frontend
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Khong decode duoc anh")
    return img


def _log_search(payload: dict[str, Any]) -> str | None:
    """Insert vao bang search_logs. Tra ve log_id."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO search_logs (
                    feature_version, top_k, result_video_ids, result_scores,
                    response_time_ms
                ) VALUES (%s, %s, %s::uuid[], %s, %s)
                RETURNING id;""",
                (
                    "v1",
                    payload["top_k"],
                    payload["video_ids"],
                    payload["scores"],
                    payload["response_time_ms"],
                ),
            )
            return str(cur.fetchone()[0])
    except Exception as e:
        # Log error nhung khong fail request
        print(f"[warn] Khong log duoc search: {e}")
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    idx = app.state.index
    return {
        "status": "ok",
        "n_frames": len(idx.frame_ids) if idx.color is not None else 0,
        "feature_version": idx.feature_version,
        "default_weights": DEFAULT_WEIGHTS,
    }


@app.post("/search")
async def search(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    color: float = Form(DEFAULT_WEIGHTS["color"]),
    texture: float = Form(DEFAULT_WEIGHTS["texture"]),
    shape: float = Form(DEFAULT_WEIGHTS["shape"]),
    by_video: bool = Form(True),
):
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k phai trong [1, 100]")

    total = color + texture + shape
    if total <= 0:
        raise HTTPException(status_code=400, detail="Tong weight phai > 0")
    weights = {"color": color / total, "texture": texture / total, "shape": shape / total}

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rong")
    img = _decode_image(content)

    t0 = time.perf_counter()
    # Lay top_k * 5 frame, sau group ve top_k video (1 video co the co nhieu frame)
    idx: SearchIndex = app.state.index
    raw_top = top_k * 5 if by_video else top_k
    raw_top = min(raw_top, len(idx.frame_ids))
    results = idx.search(img, top_k=raw_top, weights=weights)

    if by_video:
        videos = group_by_video(results, top_videos=top_k)
        out_results: list[dict] = videos
    else:
        out_results = [
            {
                "video_id": r.video_id, "video_name": r.video_name,
                "species": r.species, "frame_id": r.frame_id,
                "timestamp_sec": r.timestamp_sec, "score": r.score,
                "color_sim": r.color_sim, "texture_sim": r.texture_sim,
                "shape_sim": r.shape_sim,
            }
            for r in results[:top_k]
        ]

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    log_id = _log_search({
        "top_k": top_k,
        "video_ids": [r["video_id"] for r in out_results] if by_video else
                     [r["video_id"] for r in out_results],
        "scores": [r["score"] for r in out_results],
        "response_time_ms": elapsed_ms,
    })

    return {
        "log_id": log_id,
        "response_time_ms": elapsed_ms,
        "weights": weights,
        "by_video": by_video,
        "results": out_results,
    }


@app.get("/videos/{video_id}")
def get_video(video_id: str):
    """Chi tiet 1 video + danh sach frame."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, species, cloudinary_url, fps, duration_sec,
                      width, height, keyframe_count
               FROM videos WHERE id = %s;""",
            (video_id,),
        )
        v = cur.fetchone()
        if not v:
            raise HTTPException(status_code=404, detail="Video khong ton tai")

        cur.execute(
            """SELECT id, frame_index, timestamp_sec, motion_magnitude, motion_area
               FROM frames WHERE video_id = %s
               ORDER BY timestamp_sec;""",
            (video_id,),
        )
        frames = [
            {"frame_id": str(r[0]), "frame_index": r[1], "timestamp_sec": float(r[2]),
             "motion_magnitude": r[3], "motion_area": r[4]}
            for r in cur.fetchall()
        ]

    return {
        "video_id": str(v[0]),
        "name": v[1],
        "species": v[2],
        "cloudinary_url": v[3],
        "fps": v[4],
        "duration_sec": v[5],
        "width": v[6],
        "height": v[7],
        "keyframe_count": v[8],
        "frames": frames,
    }
