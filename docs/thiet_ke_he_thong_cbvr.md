# Thiết kế hệ thống — Truy vấn video bằng ảnh (CBVR)

## Thông tin dự án

| Mục | Chi tiết |
|---|---|
| Bài toán | Content-Based Video Retrieval — truy vấn video bằng ảnh tĩnh |
| Dữ liệu | ~1000 video ngắn về động vật hoang dã, độ dài trung bình >10s |
| Stack | Python (FastAPI), Supabase (PostgreSQL), numpy/opencv |
| Lưu trữ video | Cloudinary |
| Yêu cầu | Implement thuật toán từ đầu (không dùng model AI sẵn như CLIP) |
| Deadline | Báo cáo: 2 ngày — Sản phẩm: 1 tháng |

---

## Pipeline tổng thể

```
[OFFLINE — chạy 1 lần khi build database]

Video (Cloudinary URL)
        │
        ▼
Tách Keyframe (scene-aware delta histogram)
        │
        ▼
Trích xuất 4 đặc trưng mỗi frame
  ├── Color Histogram + Dominant Colors  (35%)
  ├── LBP + GLCM                         (35%)
  ├── HOG + Edge Density                 (20%)
  └── Frame Difference + Optical Motion  (10%) ← chỉ dùng để filter
        │
        ▼
Normalize từng đặc trưng về [0,1]
        │
        ▼
Concat → vector tổng hợp ~989 chiều
        │
        ▼
Lưu vào Supabase (bảng videos + bảng frames)


[ONLINE — khi user query]

Ảnh upload
        │
        ▼
Trích xuất 3 đặc trưng (Color + LBP/GLCM + HOG) — KHÔNG dùng Frame Difference
        │
        ▼
Normalize + Concat → vector query
        │
        ▼
Cosine Similarity với toàn bộ frame trong DB (~5000 frame)
        │
        ▼
Top-K frame → trả về video tương ứng + similarity score
```

---

## Phần 1 — Tách Keyframe

### Thuật toán: Scene-aware (delta histogram)

Không dùng 25%/50%/75% cố định. Lấy frame khi histogram thay đổi đáng kể **hoặc** đã đủ khoảng thời gian tối thiểu.

```python
def extract_keyframes(video_path, threshold=0.4, min_interval=2.0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_hist = None
    last_saved_time = -min_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        curr_hist = extract_color_histogram(frame)

        if prev_hist is not None:
            diff = 1 - cosine_similarity(curr_hist, prev_hist)
            scene_changed = diff > threshold
            interval_ok = (current_time - last_saved_time) >= min_interval

            if scene_changed or interval_ok:
                frames.append((current_time, frame))
                last_saved_time = current_time

        prev_hist = curr_hist

    cap.release()
    return frames
```

**Tại sao không dùng 25/50/75%?**
- Video 10s → chỉ lấy 3 frame, dễ bỏ sót nội dung chính
- Scene-aware đảm bảo lấy frame tại các điểm chuyển cảnh quan trọng
- Kết hợp min_interval đảm bảo coverage đều đặn

**Ước tính:** ~1000 video × ~5 frame/video = ~5000 frame tổng

---

## Phần 2 — Các thuật toán trích xuất đặc trưng

### Đặc trưng 1: Color Histogram + Dominant Colors (35%)

**Mục đích:** Nắm bắt phân phối màu sắc tổng thể và các màu chủ đạo — phân biệt loài qua màu lông/da (sư tử vàng nâu, voi xám, ngựa vằn đen trắng...)

#### Color Histogram (HSV)

```python
def extract_color_histogram(image_bgr, bins=32):
    # Dùng HSV thay vì RGB vì tách biệt màu sắc (H) khỏi độ sáng (V)
    # → robust hơn với thay đổi ánh sáng ngoài trời
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist_h = np.histogram(hsv[:,:,0], bins=bins, range=(0, 180))[0]
    hist_s = np.histogram(hsv[:,:,1], bins=bins, range=(0, 256))[0]
    hist_v = np.histogram(hsv[:,:,2], bins=bins, range=(0, 256))[0]

    feature = np.concatenate([hist_h, hist_s, hist_v])  # vector 96 chiều
    return feature / (feature.sum() + 1e-7)             # normalize
```

**Tại sao HSV?** RGB trộn lẫn màu sắc và độ sáng → cùng con vật chụp ban ngày vs chiều tối cho histogram khác nhau. HSV tách H (màu thuần) ra riêng nên ổn định hơn.

#### Dominant Colors (K-means, k=5)

```python
def extract_dominant_colors(image_bgr, k=5):
    # Reshape ảnh thành list pixels
    pixels = image_bgr.reshape(-1, 3).astype(np.float32)

    # K-means tìm k màu trung tâm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
                                     cv2.KMEANS_RANDOM_CENTERS)

    # Tính tỉ lệ xuất hiện của mỗi màu
    counts = np.bincount(labels.flatten())
    ratios = counts / counts.sum()

    # Flatten: [B1,G1,R1, ratio1, B2,G2,R2, ratio2, ...] → 20 chiều
    feature = []
    for center, ratio in zip(centers, ratios):
        feature.extend(center / 255.0)  # normalize màu về [0,1]
        feature.append(ratio)
    return np.array(feature)  # vector 20 chiều
```

**Output Color tổng:** concat(histogram_96d, dominant_20d) = **116 chiều**

---

### Đặc trưng 2: LBP + GLCM (35%)

**Mục đích:** Nắm bắt cấu trúc bề mặt (texture) — vằn hổ, vảy cá, lông xù sư tử, da trơn cá heo

#### LBP — Local Binary Pattern

```python
def extract_lbp(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)

    # Với mỗi pixel trung tâm, so sánh với 8 pixel xung quanh
    # Tạo chuỗi bit nhị phân → chuyển sang số nguyên 0-255
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray[i, j]
            code = 0
            for bit, (di, dj) in enumerate(neighbors):
                if gray[i+di, j+dj] >= center:
                    code |= (1 << bit)
            lbp[i-1, j-1] = code

    # Dùng Uniform LBP: chỉ giữ pattern có ≤2 transitions (robust hơn với noise)
    hist = np.zeros(59)  # 58 uniform patterns + 1 non-uniform
    for val in lbp.flatten():
        binary = format(val, '08b')
        transitions = sum(binary[i] != binary[i-1] for i in range(1, 8))
        transitions += (binary[7] != binary[0])
        if transitions <= 2:
            hist[val % 58] += 1
        else:
            hist[58] += 1

    return hist / (hist.sum() + 1e-7)  # vector 59 chiều (normalized)
```

**Tại sao Uniform LBP?** LBP thường rất nhạy với thay đổi ánh sáng — video ngoài trời ánh sáng thay đổi liên tục. Uniform LBP chỉ giữ 59 pattern ổn định nhất, giảm nhiễu đáng kể.

#### GLCM — Gray Level Co-occurrence Matrix

```python
def compute_glcm(gray, levels=8):
    # Quantize về levels mức để giảm kích thước ma trận
    gray_q = (gray / (256 / levels)).astype(int).clip(0, levels-1)
    glcm = np.zeros((levels, levels))

    # Đếm cặp pixel (i, j) nằm cạnh nhau theo hướng ngang
    for i in range(gray_q.shape[0]):
        for j in range(gray_q.shape[1] - 1):
            glcm[gray_q[i,j], gray_q[i,j+1]] += 1

    # Symmetrize (cộng với transpose)
    glcm = (glcm + glcm.T) / 2
    return glcm / (glcm.sum() + 1e-7)  # normalize

def extract_glcm_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    glcm = compute_glcm(gray)

    i_idx, j_idx = np.mgrid[0:glcm.shape[0], 0:glcm.shape[1]]

    contrast    = np.sum(glcm * (i_idx - j_idx)**2)
    energy      = np.sum(glcm**2)
    homogeneity = np.sum(glcm / (1 + np.abs(i_idx - j_idx)))
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    std_i = np.sqrt(np.sum(glcm * (i_idx - mu_i)**2))
    std_j = np.sqrt(np.sum(glcm * (j_idx - mu_j)**2))
    correlation = np.sum(glcm * (i_idx - mu_i) * (j_idx - mu_j)) / (std_i * std_j + 1e-7)

    return np.array([contrast, energy, homogeneity, correlation])  # vector 4 chiều
```

**Ý nghĩa 4 chỉ số GLCM:**

| Chỉ số | Cao | Thấp |
|---|---|---|
| Contrast | Texture gồ ghề (vằn sắc nét) | Texture mịn (da trơn) |
| Energy | Texture đồng đều, lặp lại | Texture ngẫu nhiên |
| Homogeneity | Pixel liền kề giống nhau | Pixel liền kề khác nhau nhiều |
| Correlation | Tương quan tuyến tính cao | Ngẫu nhiên |

**Output Texture tổng:** concat(lbp_59d, glcm_4d) = **63 chiều**

---

### Đặc trưng 3: HOG + Edge Density (20%)

**Mục đích:** Nắm bắt hình dạng tổng thể và silhouette của động vật — dáng 4 chân, tỉ lệ thân, đường viên cơ thể

#### HOG — Histogram of Oriented Gradients

```python
def resize_with_padding(image, target=(64, 64)):
    # QUAN TRỌNG: Dùng padding thay vì stretch để tránh méo hình
    h, w = image.shape[:2]
    scale = min(target[0]/h, target[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h = target[0] - new_h
    pad_w = target[1] - new_w
    padded = cv2.copyMakeBorder(resized,
                                 pad_h//2, pad_h - pad_h//2,
                                 pad_w//2, pad_w - pad_w//2,
                                 cv2.BORDER_CONSTANT, value=0)
    return padded

def extract_hog(image_bgr, resize=(64, 64)):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = resize_with_padding(gray, resize)

    # Tính gradient theo x và y bằng Sobel
    gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi % 180  # 0-180 độ

    # Chia thành cell 8x8, mỗi cell tính histogram 9 bin (mỗi bin = 20 độ)
    cell_size, n_bins = 8, 9
    hog_features = []

    for i in range(0, resize[0], cell_size):
        for j in range(0, resize[1], cell_size):
            cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
            cell_ang = angle[i:i+cell_size, j:j+cell_size]
            hist = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (cell_ang >= b*20) & (cell_ang < (b+1)*20)
                hist[b] = cell_mag[mask].sum()
            hog_features.append(hist)

    feature = np.array(hog_features).flatten()  # 8*8*9 = 576 chiều
    return feature / (feature.sum() + 1e-7)
```

#### Edge Density (Canny)

```python
def extract_edge_density(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Chia ảnh thành 4x4 = 16 vùng, tính mật độ cạnh mỗi vùng
    h, w = edges.shape
    cell_h, cell_w = h // 4, w // 4
    density = []
    for i in range(4):
        for j in range(4):
            cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            density.append(cell.mean() / 255.0)

    return np.array(density)  # vector 16 chiều
```

**Tại sao padding thay vì stretch?** Query image và frame video có aspect ratio khác nhau. Stretch thẳng về 64x64 làm méo hình → HOG của cùng 1 con vật ra vector khác nhau.

**Output Shape tổng:** concat(hog_576d, edge_density_16d) = **592 chiều** (→ giảm xuống ~200d nếu dùng HOG block nhỏ hơn)

---

### Đặc trưng 4: Frame Difference + Optical Motion (10%)

**⚠️ Chỉ dùng cho video frame — KHÔNG dùng khi query bằng ảnh tĩnh**

**Mục đích:** Lọc frame chất lượng khi build database — ưu tiên frame có động vật đang chuyển động (thay vì cảnh nền tĩnh)

```python
def compute_motion_features(frame_current, frame_prev):
    gray_curr = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY).astype(float)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY).astype(float)

    # Frame difference
    diff = np.abs(gray_curr - gray_prev)

    motion_magnitude = diff.mean() / 255.0          # cường độ chuyển động trung bình
    motion_area = (diff > 20).sum() / diff.size      # tỉ lệ vùng pixel thay đổi

    return np.array([motion_magnitude, motion_area])  # vector 2 chiều
```

**Cách dùng:** Khi lưu frame vào DB, lưu kèm motion score. Khi query, không đưa vào similarity computation — chỉ dùng để filter/rank nếu cần ưu tiên frame có hành động.

---

## Phần 3 — Kết hợp vector (QUAN TRỌNG)

### Bắt buộc: Normalize trước khi concat

Mỗi đặc trưng có scale khác nhau. Nếu không normalize, GLCM contrast (có thể lên hàng nghìn) sẽ lấn át tất cả đặc trưng khác.

```python
def normalize_feature(vector):
    min_v, max_v = vector.min(), vector.max()
    return (vector - min_v) / (max_v - min_v + 1e-7)

def extract_all_features(image_bgr):
    color_hist  = normalize_feature(extract_color_histogram(image_bgr))   # 96d
    dominant    = normalize_feature(extract_dominant_colors(image_bgr))    # 20d
    lbp         = normalize_feature(extract_lbp(image_bgr))               # 59d
    glcm        = normalize_feature(extract_glcm_features(image_bgr))     # 4d
    hog         = normalize_feature(extract_hog(image_bgr))               # 576d
    edge        = normalize_feature(extract_edge_density(image_bgr))      # 16d

    # Weighted concat theo trọng số thiết kế
    w_color = 0.35
    w_texture = 0.35
    w_shape = 0.20

    combined = np.concatenate([
        color_hist * w_color * 0.7,   # 70% của color weight
        dominant   * w_color * 0.3,   # 30% của color weight
        lbp        * w_texture * 0.9, # 90% của texture weight
        glcm       * w_texture * 0.1, # 10% của texture weight
        hog        * w_shape * 0.8,   # 80% của shape weight
        edge       * w_shape * 0.2,   # 20% của shape weight
    ])
    return combined
```

### Cosine Similarity (implement thủ công)

```python
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / (norm + 1e-7)  # 1.0 = giống hệt, 0.0 = khác hoàn toàn
```

---

## Phần 4 — Schema Database (Supabase)

### Bảng `videos` — metadata cơ bản

```sql
CREATE TABLE videos (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title           text,
    cloudinary_url  text NOT NULL,
    duration_sec    float,
    fps             int,
    resolution      text,          -- "1920x1080"
    animal_class    text,          -- "lion", "elephant", "zebra"...
    frame_count     int,           -- số keyframe đã extract
    created_at      timestamp DEFAULT now()
);
```

### Bảng `frames` — metadata chi tiết (unit so sánh)

```sql
CREATE TABLE frames (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id            uuid REFERENCES videos(id) ON DELETE CASCADE,
    timestamp_sec       float NOT NULL,     -- frame xuất hiện ở giây thứ mấy
    frame_url           text,               -- URL ảnh frame trên Cloudinary

    -- Feature vectors (dùng để search)
    color_histogram     float[],            -- 96 chiều
    dominant_colors     float[],            -- 20 chiều
    lbp_feature         float[],            -- 59 chiều
    glcm_feature        float[],            -- 4 chiều
    hog_feature         float[],            -- 576 chiều
    edge_density        float[],            -- 16 chiều
    combined_feature    float[],            -- vector tổng hợp (dùng để search)

    -- Motion metadata (chỉ lưu, không dùng trong similarity)
    motion_magnitude    float,
    motion_area         float,

    created_at          timestamp DEFAULT now()
);
```

**Lý do tách 2 bảng:**
- `videos`: hiển thị kết quả sau khi tìm được → cần metadata cơ bản
- `frames`: đơn vị so sánh trực tiếp với query image → cần chi tiết đầy đủ

---

## Phần 5 — Các lo ngại kỹ thuật cần lưu ý

| Mức độ | Vấn đề | Giải pháp |
|---|---|---|
| 🔴 Bắt buộc | Normalize trước khi concat | Dùng min-max normalize từng đặc trưng |
| 🔴 Bắt buộc | HOG bị méo do resize | Dùng resize_with_padding thay vì stretch |
| 🟡 Nên làm | LBP nhạy với ánh sáng ngoài trời | Dùng Uniform LBP (59 bins thay vì 256) |
| 🟡 Nên làm | Frame Difference không dùng khi query | Tách rõ logic offline vs online |
| 🟢 Optional | Color Histogram + Dominant bị overlap | Weight riêng từng phần khi concat |

---

## Phần 6 — Kế hoạch thực hiện

### Giai đoạn 1: Báo cáo (2 ngày)
- Hoàn thiện thiết kế trên giấy (tài liệu này)
- Sơ đồ pipeline tổng thể
- Mô tả thuật toán từng đặc trưng
- Schema database
- Prototype đơn giản minh họa (optional)

### Giai đoạn 2: Sản phẩm (1 tháng)

**Tuần 1:** Implement 4 đặc trưng + unit test từng hàm

**Tuần 2:** Batch job xử lý 1000 video → đẩy vector vào Supabase

**Tuần 3:** API FastAPI (`/search` nhận ảnh → trả top-K video)

**Tuần 4:** Demo web (upload ảnh → hiển thị kết quả + score), tối ưu và đánh giá

---

*Tài liệu thiết kế — Cơ sở dữ liệu đa phương tiện*
