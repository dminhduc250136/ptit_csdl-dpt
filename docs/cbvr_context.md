# CBVR — Context tổng hợp dự án

## Thông tin dự án

| Mục | Chi tiết |
|---|---|
| Bài toán | Content-Based Video Retrieval — truy vấn video bằng ảnh tĩnh |
| Dữ liệu | ~917 video về động vật hoang dã (sau làm sạch), 71 loài |
| Stack | Python (FastAPI), Supabase (PostgreSQL), numpy / opencv / skimage |
| Lưu trữ video | Cloudinary |
| Yêu cầu | Implement thuật toán từ đầu — không dùng model AI sẵn (CLIP, YOLO...) |
| Deadline | Báo cáo: 2 ngày — Sản phẩm: 1 tháng |

---

## Pipeline tổng thể

```
[OFFLINE — chạy 1 lần khi build database]

metadata_cleaned.json (917 video)
        │
        ▼
Download video từ Cloudinary URL
        │
        ▼ (nếu needs_resize = true)
Resize về 1280x720 trước khi xử lý
        │
        ▼
Tách Keyframe (scene-aware delta histogram)
  - threshold=0.4, min_interval=2s
  - Ước tính ~5 frame/video → ~4500 frame tổng
        │
        ▼
Trích xuất đặc trưng mỗi frame
  ├── Color Histogram + Dominant Colors  (weight 35%)
  ├── LBP + GLCM                         (weight 35%)
  ├── HOG + Edge Density                 (weight 20%)
  └── Frame Difference                   (không dùng trong similarity,
                                          chỉ lưu để filter/boost)
        │
        ▼
Normalize từng nhóm đặc trưng về [0,1]
        │
        ▼
Lưu từng vector riêng vào Supabase (bảng frames)


[ONLINE — khi user query]

Ảnh upload (query image)
        │
        ▼
Trích xuất 3 nhóm đặc trưng (Color + Texture + Shape)
— KHÔNG dùng Frame Difference vì query là ảnh tĩnh
        │
        ▼
Normalize + tính Weighted Similarity với từng frame trong DB
  score = 0.35 × cosine_sim(color)
        + 0.35 × cosine_sim(texture)
        + 0.20 × cosine_sim(shape)
  (tính similarity từng nhóm riêng, KHÔNG nhân weight vào vector rồi concat)
        │
        ▼
Rank → trả Top-K frame → group theo video → trả kết quả
```

---

## Dữ liệu — Trạng thái sau làm sạch

### Tổng quan

| Chỉ số | Số lượng |
|---|---|
| Video ban đầu | 943 |
| Đã xóa — video dọc (portrait) | 20 |
| Đã xóa — tỉ lệ khung hình quá lệch | 5 |
| Đã xóa — video quá dài (608s) | 1 |
| **Video còn lại** | **917** |
| Cần resize về 1280×720 khi extract | 30 |
| Số loài | 71 |
| Số loài có < 14 video | 28 |

### Nguồn dữ liệu

Hai nguồn, phân biệt qua prefix tên:
- Không có prefix (vd: `tiger_12345`) — nguồn 1, ~485 video
- Prefix `pixabay_` (vd: `pixabay_tiger_12345`) — nguồn Pixabay, ~458 video (sau lọc ~432)

### Vấn đề đã xử lý

**ID gốc bị trùng giữa các loài (53 ID, 110 record):**
Nguyên nhân: script crawl gán ID từ nguồn gốc, nhưng cùng một ID số có thể tồn tại ở nhiều loài khác nhau (ví dụ `bear_168896` và `sloth_bear_168896` là hai video khác nhau nhưng cùng ID 168896). Không phải lỗi dữ liệu thực sự vì `name` = species + id là đủ phân biệt. Giải pháp: **không dùng ID gốc làm primary key**, thay bằng UUID mới sinh khi insert DB.

**Video quá dài (meerkat_856925 = 608s):**
Nếu giữ sẽ sinh ~300 keyframe từ một video, làm lệch kết quả search. Đã xóa.

**Video dọc và tỉ lệ lệch:**
HOG và shape feature nhạy cảm với tỉ lệ khung hình. Video dọc hoặc quá lệch sẽ tạo ra vector shape không tương thích với query image thông thường. Đã xóa 25 video.

**30 video cần resize (1366×720, 960×540...):**
Tỉ lệ gần 16:9, giữ lại nhưng cần resize về 1280×720 trước khi extract. Flag `needs_resize=true` trong metadata_cleaned.json.

### File metadata sạch

File: `metadata_cleaned.json`

Cấu trúc mỗi record:
```json
{
  "uuid": "...",            // primary key dùng khi insert DB
  "name": "tiger_12345",    // tên gốc, giữ để debug
  "species": "tiger",
  "fps": 29.97,
  "frame_count": 800,
  "width": 1280,
  "height": 720,
  "duration_sec": 26.69,
  "file_size_bytes": 8175966,
  "cloudinary_url": "https://res.cloudinary.com/...",
  "needs_resize": false
}
```

---

## Các đặc trưng trích xuất

### Tổng quan 4 nhóm

| Nhóm | Thuật toán | Đo cái gì | Output | Dùng cho | Weight |
|---|---|---|---|---|---|
| Color | HSV Histogram + K-means Dominant Colors | Màu lông/da đặc trưng của loài | 96 + 20 = 116 chiều | Query + Frame | 35% |
| Texture | Uniform LBP + GLCM | Cấu trúc bề mặt: vằn, vảy, lông xù, da trơn | 59 + 4 = 63 chiều | Query + Frame | 35% |
| Shape | HOG + Edge Density 4×4 | Hình dạng tổng thể, silhouette, body structure | 576 + 16 = 592 chiều | Query + Frame | 20% |
| Temporal | Frame Difference | Chuyển động giữa 2 frame liên tiếp | 2 chiều | Chỉ Frame | Không dùng trong similarity |

> Lưu ý: HOG 576 chiều quá lớn so với các nhóm khác (chiếm 75% tổng vector). Nên reduce bằng cách tăng `cell_size=16` → HOG về 144 chiều. Tổng vector lúc đó: 116 + 63 + 144 + 16 = **339 chiều**.

### Chi tiết từng nhóm

#### Color (116 chiều)

**HSV Histogram (96 chiều):**
- Convert BGR → HSV
- Histogram 32 bins mỗi kênh H, S, V → concat → 96 chiều
- Dùng HSV thay RGB vì tách màu sắc (H) khỏi độ sáng (V) → ổn định hơn với thay đổi ánh sáng ngoài trời

**Dominant Colors — K-means k=5 (20 chiều):**
- Reshape ảnh thành list pixels → K-means 5 cụm
- Output: [B, G, R, ratio] × 5 = 20 chiều
- Sort centers theo ratio giảm dần (tránh output không deterministic)

#### Texture (63 chiều)

**Uniform LBP (59 chiều):**
- Mỗi pixel so sánh với 8 pixel xung quanh → chuỗi 8 bit → số 0–255
- Chỉ giữ 58 "uniform pattern" (≤2 transitions) + 1 bucket non-uniform = 59 chiều
- Dùng Uniform thay full 256 bins vì ổn định hơn với nhiễu ánh sáng ngoài trời
- Nên dùng `skimage.feature.local_binary_pattern` thay vì vòng lặp Python (nhanh hơn ~100×)

**GLCM (4 chiều):**
- Gray Level Co-occurrence Matrix — đếm cặp pixel liền kề
- 4 chỉ số: contrast, energy, homogeneity, correlation
- Nên dùng `skimage.feature.graycomatrix` thay vòng lặp Python

#### Shape (592 chiều → nên giảm về 160 chiều)

**HOG (576 chiều → nên giảm về 144 chiều):**
- Resize ảnh về 64×64 dùng padding (không stretch) để tránh méo hình
- Chia thành cell, tính histogram gradient 9 bins mỗi cell
- `cell_size=16` thay vì 8 → 4×4×9 = 144 chiều, cân đối hơn

**Edge Density 4×4 (16 chiều):**
- Canny edge detection → chia 4×4 vùng → tính mật độ cạnh mỗi vùng
- Capture phân bố spatial của đường viền cơ thể

#### Temporal (2 chiều — chỉ lưu, không dùng trong similarity)

- `motion_magnitude`: mean(|frame_t − frame_{t-1}|) / 255
- `motion_area`: tỉ lệ pixel thay đổi > ngưỡng 20
- Nullable với frame đầu tiên của video (không có frame trước)
- Dùng để: boost frame có chuyển động trong kết quả nếu cần

### Cách tính similarity đúng

```python
# ĐÚNG — tính cosine similarity từng nhóm, rồi weighted sum
WEIGHTS = {'color': 0.35, 'texture': 0.35, 'shape': 0.20}
DIMS    = {'color': 116,  'texture': 63,   'shape': 160}

def weighted_similarity(query_vec, frame_vec):
    score, offset = 0.0, 0
    for group in ['color', 'texture', 'shape']:
        dim = DIMS[group]
        q = query_vec[offset:offset+dim]
        f = frame_vec[offset:offset+dim]
        score += WEIGHTS[group] * cosine_similarity(q, f)
        offset += dim
    return score

# SAI — nhân weight vào vector rồi mới tính cosine (vector dài vẫn thống trị)
# combined = np.concatenate([color * 0.35, texture * 0.35, hog * 0.20])
# cosine_similarity(q_combined, f_combined)  ← HOG 576d chiếm ~78% dot product
```

---

## Schema Database (Supabase / PostgreSQL)

### Bảng `videos`

```sql
CREATE TABLE videos (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name             text NOT NULL,           -- tên gốc vd "tiger_12345", để debug
    species          text NOT NULL,           -- nhãn loài, dùng để evaluate kết quả
    cloudinary_url   text NOT NULL,
    fps              float,
    duration_sec     float,
    frame_count      int,
    width            int,
    height           int,
    needs_resize     boolean DEFAULT false,   -- flag cần resize khi extract
    file_size_bytes  bigint,
    keyframe_count   int,                     -- null = chưa xử lý, cập nhật sau khi extract
    processed_at     timestamp,              -- null = chưa extract, dùng để resume batch job
    created_at       timestamp DEFAULT now()
);
```

**Lý do từng trường:**
- `name` — giữ lại để tra cứu và debug, không dùng làm key
- `species` — nhãn chính để đánh giá precision/recall sau này
- `needs_resize` — 30 video cần resize, batch job đọc flag này để xử lý đúng
- `keyframe_count` + `processed_at` — dùng để theo dõi tiến độ batch job, resume nếu bị interrupt

### Bảng `frames`

```sql
CREATE TABLE frames (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id          uuid NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    timestamp_sec     float NOT NULL,    -- giây thứ mấy trong video, dùng khi play
    frame_index       int,               -- frame thứ mấy trong video gốc, để debug

    -- Feature vectors (lưu riêng từng nhóm để linh hoạt đổi weight sau này)
    color_histogram   float[],           -- 96 chiều
    dominant_colors   float[],           -- 20 chiều
    lbp_feature       float[],           -- 59 chiều
    glcm_feature      float[],           -- 4 chiều
    hog_feature       float[],           -- 144 chiều (sau khi reduce cell_size=16)
    edge_density      float[],           -- 16 chiều

    -- Temporal (lưu để dùng sau, không tham gia similarity)
    motion_magnitude  float,             -- null nếu là frame đầu tiên của video
    motion_area       float,

    created_at        timestamp DEFAULT now()
);

CREATE INDEX idx_frames_video_id ON frames(video_id);
```

**Lý do lưu từng vector riêng thay vì một vector tổng hợp:**
Nếu sau này muốn thay weight (ví dụ color 40% thay vì 35%), không cần chạy lại extract — chỉ cần tính lại similarity với weight mới. Nếu gộp thành một vector thì phải extract lại toàn bộ.

### Bảng `search_logs`

```sql
CREATE TABLE search_logs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_image_url   text,             -- URL ảnh query người dùng upload
    top_k             int,              -- số kết quả yêu cầu
    result_video_ids  uuid[],           -- danh sách video trả về theo thứ tự score
    result_scores     float[],          -- score tương ứng
    response_time_ms  int,              -- thời gian xử lý, phát hiện bottleneck
    created_at        timestamp DEFAULT now()
);
```

**Lý do có bảng này:**
Không ảnh hưởng đến kết quả tìm kiếm, nhưng cần để đánh giá hệ thống — xem query nào cho kết quả tốt/xấu, response time có ổn định không, và sau này tính precision nếu có ground truth.

### Quan hệ

```
videos (1) ──────< frames (nhiều)    video_id FK, ON DELETE CASCADE
videos      <──── search_logs         lưu uuid[] array, không FK cứng
                                      (log không cần cascade khi xóa video)
```

---

## Các vấn đề kỹ thuật đã ghi nhận (chưa sửa)

| Mức | Vấn đề | Giải pháp đề xuất |
|---|---|---|
| 🔴 | Weighted concat sai — nhân weight vào vector rồi tính cosine | Tính cosine similarity từng nhóm riêng rồi mới weighted sum |
| 🔴 | LBP và GLCM dùng vòng lặp Python — rất chậm | Thay bằng `skimage.feature` (vectorized, ~100× nhanh hơn) |
| 🟡 | HOG 576 chiều chiếm 75% tổng vector — mất cân bằng | Tăng `cell_size=16` → 144 chiều |
| 🟡 | Normalize min-max theo từng ảnh riêng lẻ — mất thông tin tương đối | Fit scaler trên toàn bộ DB sau khi extract xong |
| 🟡 | Keyframe dùng `or` logic — có thể lấy quá nhiều frame với video dài | Thêm `max_frames` per video |
| 🟢 | Dominant Colors không sort theo ratio | Sort centers theo `argsort(-counts)` trước khi flatten |
| 🟢 | Thiếu resize input ảnh query | Thêm `cv2.resize(img, (256, 256))` đầu hàm extract |

---

## Kế hoạch thực hiện

### Giai đoạn 1 — Báo cáo (2 ngày)
- Tài liệu thiết kế (file này)
- Sơ đồ pipeline
- Schema database
- Prototype minh họa (optional)

### Giai đoạn 2 — Sản phẩm (1 tháng)

**Tuần 1:** Implement 4 hàm extract + unit test từng hàm. Ưu tiên sửa 2 lỗi đỏ trước khi viết thêm code.

**Tuần 2:** Batch job — đọc `metadata_cleaned.json`, download video từ Cloudinary, extract keyframe + feature, insert vào Supabase. Xử lý resume (dựa vào `processed_at IS NULL`).

**Tuần 3:** API FastAPI — endpoint `/search` nhận ảnh, trích feature, tính weighted similarity với toàn bộ frames trong DB, trả top-K video kèm score và timestamp.

**Tuần 4:** Demo web — upload ảnh, hiển thị kết quả với thumbnail frame và nút play đúng điểm. Đánh giá precision theo loài.

---

## Câu hỏi còn mở

- Có dùng **pgvector** để index và tìm kiếm nhanh không, hay tính similarity thủ công bằng Python? (Ảnh hưởng đến cách lưu vector trong DB)
- Số lượng loài có < 14 video (28/71 loài) — có ảnh hưởng đến yêu cầu đánh giá không?
