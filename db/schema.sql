-- ============================================================
-- CBVR Database Schema (PostgreSQL / Supabase)
-- ============================================================
-- Stack: Supabase (PostgreSQL), float[] cho feature vector
-- Lưu ý: chưa dùng pgvector — sau này muốn bật ANN search
--        thì migrate từng cột float[] -> vector(N) là đủ.
-- ============================================================


-- ------------------------------------------------------------
-- 1) videos — metadata cấp video
-- ------------------------------------------------------------
-- Một row = một video gốc trên Cloudinary.
-- Giữ name gốc để debug, nhưng primary key dùng UUID mới sinh
-- vì id gốc của nguồn crawl bị trùng giữa các loài.
CREATE TABLE videos (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Định danh & phân loại
    name              text NOT NULL,           -- vd "tiger_12345" hoặc "pixabay_tiger_12345"
    source_id         text,                    -- id gốc từ nguồn crawl (chỉ để debug, không unique)
    species           text NOT NULL,           -- nhãn loài → dùng evaluate precision/recall

    -- Lưu trữ
    cloudinary_url    text NOT NULL,

    -- Thuộc tính kỹ thuật của video
    fps               real,
    frame_count       integer,
    width             integer,
    height            integer,
    duration_sec      real,
    file_size_bytes   bigint,

    -- Cờ tiền xử lý
    needs_resize      boolean NOT NULL DEFAULT false,
                      -- 30 video có resolution lệch chuẩn (1366x720, 960x540...)
                      -- → batch job đọc cờ này để resize về 1280x720 trước khi extract

    -- Trạng thái pipeline (resume-friendly)
    keyframe_count    integer,                 -- NULL = chưa extract xong
    processed_at      timestamptz,             -- NULL = chưa xử lý

    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_videos_name      ON videos(name);
CREATE        INDEX idx_videos_species   ON videos(species);
CREATE        INDEX idx_videos_pending   ON videos(processed_at)
                                          WHERE processed_at IS NULL;


-- ------------------------------------------------------------
-- 2) frames — metadata cấp keyframe
-- ------------------------------------------------------------
-- Một row = một keyframe của một video.
-- KHÔNG chứa feature vector → bảng nhẹ, JOIN nhanh.
-- Strategy keyframe lưu trong cột `extraction_method` để
-- có thể tồn tại song song nhiều phương pháp (25/50/75 cũ
-- và scene-aware mới).
CREATE TABLE frames (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id           uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,

    -- Vị trí trong video
    frame_index        integer NOT NULL,       -- frame thứ mấy trong video gốc
    timestamp_sec      real    NOT NULL,       -- giây thứ mấy → dùng khi play đúng điểm

    -- Phương pháp tách keyframe
    extraction_method  text NOT NULL DEFAULT 'scene_aware',
                       -- 'fixed_25_50_75'  → strategy cũ (legacy data)
                       -- 'scene_aware'     → delta histogram theo docs
    position_percent   smallint,               -- chỉ có giá trị với fixed_25_50_75
                                                -- (25, 50, 75); NULL với scene_aware

    -- Temporal features (chỉ lưu để filter/boost, KHÔNG dùng trong similarity)
    motion_magnitude   real,                   -- NULL nếu là frame đầu tiên
    motion_area        real,

    -- Lưu trữ ảnh keyframe (optional — để hiển thị thumbnail)
    image_url          text,

    created_at         timestamptz NOT NULL DEFAULT now(),

    UNIQUE (video_id, frame_index, extraction_method)
);

CREATE INDEX idx_frames_video_id ON frames(video_id);


-- ------------------------------------------------------------
-- 3) frame_features — feature vectors (TÁCH RA BẢNG RIÊNG)
-- ------------------------------------------------------------
-- Lý do tách:
--  - Bảng frames nhẹ → query metadata nhanh (list keyframe của 1 video,
--    đếm theo loài, hiển thị UI...).
--  - Cho phép versioning: cùng 1 frame có thể có nhiều phiên bản feature
--    (vd v1 = HOG 576d, v2 = HOG 144d sau khi sửa cell_size=16).
--    Khi sửa lỗi đỏ trong docs, không phải xóa data cũ.
--  - Đổi weight không cần extract lại (vẫn lưu từng vector riêng).
-- Lý do KHÔNG tách thêm thành 6 bảng:
--  - Search luôn cần đủ 3 nhóm color/texture/shape một lúc → 6 JOIN sẽ chậm.
--  - 6 cột float[] trong cùng row đọc 1 lần là tối ưu.
CREATE TABLE frame_features (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    frame_id          uuid NOT NULL REFERENCES frames(id) ON DELETE CASCADE,

    -- Versioning: cho phép tồn tại nhiều bộ feature cho cùng 1 frame
    feature_version   text NOT NULL DEFAULT 'v1',
                      -- 'v1' = baseline theo docs (HOG 576d, LBP loop python...)
                      -- 'v2' = sau khi fix lỗi đỏ (HOG 144d, skimage vectorized...)

    -- Color group (35% weight)  — concat dim = 116
    color_histogram   real[]   NOT NULL,       -- 96 chiều (HSV histogram)
    dominant_colors   real[]   NOT NULL,       -- 20 chiều (K-means k=5)

    -- Texture group (35% weight) — concat dim = 63
    lbp_feature       real[]   NOT NULL,       -- 59 chiều (Uniform LBP)
    glcm_feature      real[]   NOT NULL,       --  4 chiều (contrast, energy, homogeneity, correlation)

    -- Shape group (20% weight)  — concat dim = 160 (sau fix) hoặc 592 (baseline)
    hog_feature       real[]   NOT NULL,       -- 144 hoặc 576 chiều
    edge_density      real[]   NOT NULL,       -- 16 chiều (Canny 4x4)

    created_at        timestamptz NOT NULL DEFAULT now(),

    UNIQUE (frame_id, feature_version)
);

CREATE INDEX idx_frame_features_frame_id ON frame_features(frame_id);
CREATE INDEX idx_frame_features_version  ON frame_features(feature_version);


-- ------------------------------------------------------------
-- 4) search_logs — đánh giá chất lượng & latency
-- ------------------------------------------------------------
-- Không ảnh hưởng search, chỉ để monitoring & evaluation.
CREATE TABLE search_logs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_image_url   text,
    feature_version   text,                    -- bộ feature dùng khi search
    top_k             integer,
    result_video_ids  uuid[],
    result_scores     real[],
    response_time_ms  integer,
    created_at        timestamptz NOT NULL DEFAULT now()
);


-- ============================================================
-- Quan hệ
--
-- videos (1) ──< frames (n) ──< frame_features (n)
-- videos    <── search_logs.result_video_ids[] (soft ref, không FK)
-- ============================================================
