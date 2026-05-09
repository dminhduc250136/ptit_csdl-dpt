"""Similarity search: load all frame features tu DB, tinh weighted cosine.

Kien truc:
  - SearchIndex: load 1 lan, giu trong RAM (5000 frame * 771d * 4 bytes ~= 16 MB).
  - search(query_image_bgr, top_k): extract feature -> cosine per group -> weighted sum -> top-K.

Cosine per group, KHONG nhan weight vao vector roi concat
(theo loi do trong docs/cbvr_context.md).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.db import get_conn
from src.features import extract_all

DEFAULT_WEIGHTS = {"color": 0.0, "texture": 0.9, "shape": 0.1}
# Tuned tu grid search 80 query (scripts/tune_weights):
# best p@10 = 0.534 vs default 35/35/20 -> 0.496.
# Color bi nhieu vi histogram chung background outdoor (co/troi),
# texture (LBP+GLCM) la discriminator manh nhat giua cac loai.

# Mapping nhom -> cac cot vector trong DB
GROUP_COLUMNS = {
    "color": ["color_histogram", "dominant_colors"],
    "texture": ["lbp_feature", "glcm_feature"],
    "shape": ["hog_feature", "edge_density"],
}


@dataclass
class SearchResult:
    video_id: str
    video_name: str
    species: str
    frame_id: str
    timestamp_sec: float
    score: float
    color_sim: float
    texture_sim: float
    shape_sim: float


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize tung row -> dot product = cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return matrix / norms


# Identity helper - giu chung interface cho experimentation sau nay.
# Test bo bin flat (0, 8, 58) thay p@10 GIAM 0.534 -> 0.429,
# nen revert. Bin "flat dominate" hoa ra van mang signal phan biet loai
# (vd: jellyfish nen nuoc co % flat khac voi voi dang chuyen dong).
def _strip_flat_bin(lbp: np.ndarray) -> np.ndarray:
    return np.asarray(lbp, dtype=np.float32)


class SearchIndex:
    """In-memory index. Load 1 lan, query nhieu lan."""

    def __init__(self, feature_version: str = "v1"):
        self.feature_version = feature_version
        self.frame_ids: list[str] = []
        self.video_ids: list[str] = []
        self.video_names: list[str] = []
        self.species: list[str] = []
        self.timestamps: list[float] = []

        # Mot ma tran (n_frames, dim) cho moi nhom, da L2-normalized
        self.color: np.ndarray | None = None
        self.texture: np.ndarray | None = None
        self.shape: np.ndarray | None = None

    def load(self) -> "SearchIndex":
        sql = """
            SELECT
                ff.frame_id, fr.video_id, v.name, v.species, fr.timestamp_sec,
                ff.color_histogram, ff.dominant_colors,
                ff.lbp_feature, ff.glcm_feature,
                ff.hog_feature, ff.edge_density
            FROM frame_features ff
            JOIN frames fr ON fr.id = ff.frame_id
            JOIN videos v  ON v.id = fr.video_id
            WHERE ff.feature_version = %s
            ORDER BY ff.frame_id;
        """
        t0 = time.perf_counter()
        # Dung named cursor (server-side) de stream tung batch -> tranh
        # statement_timeout cua Supabase pooler khi fetch 5000+ row 1 lan.
        rows = []
        with get_conn() as conn:
            with conn.cursor(name="search_index_load") as cur:
                cur.itersize = 500
                cur.execute(sql, (self.feature_version,))
                for batch in iter(lambda: cur.fetchmany(500), []):
                    rows.extend(batch)

        if not rows:
            raise RuntimeError(
                f"Khong co frame nao voi feature_version={self.feature_version}"
            )

        color_list, texture_list, shape_list = [], [], []
        for r in rows:
            (frame_id, video_id, vname, species, ts,
             c_hist, c_dom, lbp, glcm, hog, edge) = r
            self.frame_ids.append(str(frame_id))
            self.video_ids.append(str(video_id))
            self.video_names.append(vname)
            self.species.append(species)
            self.timestamps.append(float(ts))

            # Fix LBP saturation: bin 8 (uniform-flat) chiem 70-90% mass
            # va bin 58 (non-uniform) cung lon -> 2 bin nay an het signal.
            # Set 0 + renormalize 57 bin con lai de cosine phan biet duoc cac loai.
            lbp = _strip_flat_bin(np.asarray(lbp, dtype=np.float32))

            color_list.append(np.concatenate([c_hist, c_dom]).astype(np.float32))
            texture_list.append(np.concatenate([lbp, glcm]).astype(np.float32))
            shape_list.append(np.concatenate([hog, edge]).astype(np.float32))

        self.color = _normalize_rows(np.stack(color_list))
        self.texture = _normalize_rows(np.stack(texture_list))
        self.shape = _normalize_rows(np.stack(shape_list))

        print(
            f"[SearchIndex] Loaded {len(rows)} frames "
            f"(color={self.color.shape}, texture={self.texture.shape}, "
            f"shape={self.shape.shape}) in {time.perf_counter()-t0:.2f}s"
        )
        return self

    def search(
        self,
        query_image_bgr: np.ndarray,
        top_k: int = 10,
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        if self.color is None:
            raise RuntimeError("Index chua load - goi .load() truoc")
        w = weights or DEFAULT_WEIGHTS

        feats = extract_all(query_image_bgr)
        # Apply LBP fix tuong tu khi load DB de ket qua nhat quan
        lbp_fixed = _strip_flat_bin(feats["lbp_feature"])
        q_color = np.concatenate([feats["color_histogram"], feats["dominant_colors"]])
        q_texture = np.concatenate([lbp_fixed, feats["glcm_feature"]])
        q_shape = np.concatenate([feats["hog_feature"], feats["edge_density"]])

        # Normalize query, dot product = cosine similarity
        q_color = q_color / (np.linalg.norm(q_color) + 1e-9)
        q_texture = q_texture / (np.linalg.norm(q_texture) + 1e-9)
        q_shape = q_shape / (np.linalg.norm(q_shape) + 1e-9)

        sim_color = self.color @ q_color
        sim_texture = self.texture @ q_texture
        sim_shape = self.shape @ q_shape

        score = (
            w["color"] * sim_color
            + w["texture"] * sim_texture
            + w["shape"] * sim_shape
        )

        # Top-K theo score
        top_idx = np.argpartition(-score, min(top_k, len(score) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-score[top_idx])]

        return [
            SearchResult(
                video_id=self.video_ids[i],
                video_name=self.video_names[i],
                species=self.species[i],
                frame_id=self.frame_ids[i],
                timestamp_sec=self.timestamps[i],
                score=float(score[i]),
                color_sim=float(sim_color[i]),
                texture_sim=float(sim_texture[i]),
                shape_sim=float(sim_shape[i]),
            )
            for i in top_idx
        ]


def group_by_video(
    results: Sequence[SearchResult], top_videos: int = 5
) -> list[dict]:
    """Gop ket qua frame ve video. Moi video lay frame co score cao nhat."""
    seen: dict[str, dict] = {}
    for r in results:
        if r.video_id not in seen or r.score > seen[r.video_id]["score"]:
            seen[r.video_id] = {
                "video_id": r.video_id,
                "video_name": r.video_name,
                "species": r.species,
                "best_frame_id": r.frame_id,
                "best_timestamp_sec": r.timestamp_sec,
                "score": r.score,
                "color_sim": r.color_sim,
                "texture_sim": r.texture_sim,
                "shape_sim": r.shape_sim,
            }
    out = sorted(seen.values(), key=lambda x: -x["score"])
    return out[:top_videos]
