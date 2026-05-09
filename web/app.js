// CBVR Demo — vanilla JS, no framework

const API = "http://127.0.0.1:8000";

// ---------- DOM refs ----------
const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput");
const dropzone  = $("dropzone");
const dzEmpty   = $("dzEmpty");
const dzPreview = $("dzPreview");
const searchBtn = $("searchBtn");
const status    = $("status");
const results   = $("results");
const health    = $("health");
const topK      = $("topK");
const byVideo   = $("byVideo");
const sliders   = {
  color:   { input: $("wColor"),   val: $("wColorVal") },
  texture: { input: $("wTexture"), val: $("wTextureVal") },
  shape:   { input: $("wShape"),   val: $("wShapeVal") },
};

// ---------- State ----------
let currentFile = null;

// ---------- Health check ----------
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    health.textContent = `${d.n_frames.toLocaleString()} frames · ${d.feature_version}`;
    health.classList.add("ok");
  } catch (e) {
    health.textContent = "API offline (cần chạy uvicorn src.api:app)";
    health.classList.add("bad");
  }
}

// ---------- File handling ----------
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("File không phải ảnh", true);
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  dzPreview.src = url;
  dzPreview.hidden = false;
  dzEmpty.hidden = true;
  searchBtn.disabled = false;
  setStatus(`Đã chọn: ${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
}

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

// Paste image from clipboard
window.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items || [];
  for (const it of items) {
    if (it.type.startsWith("image/")) {
      setFile(it.getAsFile());
      break;
    }
  }
});

// ---------- Sliders & presets ----------
function updateSliderLabels() {
  for (const k in sliders) {
    sliders[k].val.textContent = parseFloat(sliders[k].input.value).toFixed(2);
  }
}
for (const k in sliders) {
  sliders[k].input.addEventListener("input", updateSliderLabels);
}
document.querySelectorAll(".presets button").forEach((btn) => {
  btn.addEventListener("click", () => {
    sliders.color.input.value   = btn.dataset.c;
    sliders.texture.input.value = btn.dataset.t;
    sliders.shape.input.value   = btn.dataset.s;
    updateSliderLabels();
  });
});

// ---------- Search ----------
searchBtn.addEventListener("click", search);

function setStatus(msg, isError = false) {
  status.textContent = msg;
  status.classList.toggle("error", isError);
}

async function search() {
  if (!currentFile) return;
  searchBtn.disabled = true;
  setStatus("Đang tìm kiếm...");

  const fd = new FormData();
  fd.append("image", currentFile);
  fd.append("top_k", topK.value);
  fd.append("color",   sliders.color.input.value);
  fd.append("texture", sliders.texture.input.value);
  fd.append("shape",   sliders.shape.input.value);
  fd.append("by_video", byVideo.checked);

  try {
    const t0 = performance.now();
    const r = await fetch(`${API}/search`, { method: "POST", body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const d = await r.json();
    const dt = (performance.now() - t0).toFixed(0);
    setStatus(`${d.results.length} kết quả · server ${d.response_time_ms}ms · total ${dt}ms`);
    renderResults(d.results, byVideo.checked);
  } catch (e) {
    setStatus(`Lỗi: ${e.message}`, true);
  } finally {
    searchBtn.disabled = false;
  }
}

// ---------- Render ----------
function fmtTime(s) {
  if (s == null) return "--:--";
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${ss}`;
}

function renderResults(items, isByVideo) {
  if (!items.length) {
    results.innerHTML = `<p class="empty">Không có kết quả.</p>`;
    return;
  }

  results.innerHTML = "";
  items.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "card";

    // Lay timestamp + url qua API /videos/{id}.
    // De render nhanh, set placeholder roi load metadata async.
    const ts = isByVideo ? r.best_timestamp_sec : r.timestamp_sec;
    card.innerHTML = `
      <div class="card-thumb">
        <span class="card-rank">#${i + 1}</span>
        <span class="card-time">${fmtTime(ts)}</span>
        <video preload="metadata" muted playsinline></video>
      </div>
      <div class="card-info">
        <div class="card-species">${r.species}</div>
        <div class="card-name" title="${r.video_name}">${r.video_name}</div>
        <div class="card-scores">
          <span class="score-main">score: ${r.score.toFixed(4)}</span>
          <span>color: ${r.color_sim.toFixed(3)}</span>
          <span>texture: ${r.texture_sim.toFixed(3)}</span>
          <span>shape: ${r.shape_sim.toFixed(3)}</span>
        </div>
      </div>
    `;
    results.appendChild(card);

    // Async load video URL & seek thumbnail to keyframe timestamp
    fetch(`${API}/videos/${r.video_id}`)
      .then((res) => res.json())
      .then((d) => {
        const v = card.querySelector("video");
        v.src = d.cloudinary_url;
        // Seek to the matching keyframe timestamp
        v.addEventListener("loadedmetadata", () => {
          try { v.currentTime = ts || 0; } catch (e) {}
        }, { once: true });
        // Click -> open modal
        card.addEventListener("click", () => openModal(d, ts, r));
      });
  });
}

// ---------- Modal ----------
const modal      = $("modal");
const modalClose = $("modalClose");
const modalTitle = $("modalTitle");
const modalMeta  = $("modalMeta");
const modalVideo = $("modalVideo");

function openModal(video, ts, result) {
  modalTitle.textContent = video.species;
  modalMeta.textContent =
    `${video.name} · ${video.duration_sec?.toFixed(1)}s · ` +
    `${video.fps} fps · ${video.width}x${video.height} · ` +
    `score ${result.score.toFixed(4)}`;
  modalVideo.src = video.cloudinary_url;
  modalVideo.addEventListener("loadedmetadata", () => {
    try { modalVideo.currentTime = ts || 0; } catch (e) {}
    modalVideo.play().catch(() => {});
  }, { once: true });
  modal.hidden = false;
}

function closeModal() {
  modal.hidden = true;
  modalVideo.pause();
  modalVideo.src = "";
}
modalClose.addEventListener("click", closeModal);
modal.querySelector(".modal-bg").addEventListener("click", closeModal);
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !modal.hidden) closeModal();
});

// ---------- Init ----------
checkHealth();
updateSliderLabels();
