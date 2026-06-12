import os, pathlib
import unicodedata
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CNN_MODEL_PATH  = "food_model.h5"    
YOLO_MODEL_PATH = "egg.pt"          
CLASS_NAMES_TXT = "class_names.txt"    
IMG_SIZE        = 128                 

PRICE_MAP = {
    "cơm":                   10_000,
    "đậu hũ sốt cà":        25_000,
    "cá hú kho":             30_000,
    "thịt kho trứng":        30_000,
    "thịt kho":              25_000,
    "canh chua có cá":       25_000,
    "canh chua không cá":    10_000,
    "sườn nướng":            30_000,
    "canh rau cải thảo":      7_000,
    "canh rau muống":         7_000,
    "rau xào lagim":         10_000,
    "rau xào củ sắn":        10_000,
    "rau xào đậu que":       10_000,
    "rau xào đậu đũa":       10_000,
    "trứng chiên":           25_000,
    "trứng chiên thịt":      30_000,
    "không rõ":                   0,
}


DISPLAY_NAMES = {
    "cơm":                  "Cơm trắng",
    "đậu hũ sốt cà":       "Đậu hũ sốt cà",
    "cá hú kho":            "Cá hú kho",
    "thịt kho trứng":       "Thịt kho trứng",
    "thịt kho":             "Thịt kho",
    "canh chua có cá":      "Canh chua có cá",
    "canh chua không cá":   "Canh chua không cá",
    "sườn nướng":           "Sườn nướng",
    "canh rau cải thảo":    "Canh rau cải thảo",
    "canh rau muống":       "Canh rau muống",
    "rau xào lagim":        "Rau xào lagim",
    "rau xào củ sắn":       "Rau xào củ sắn",
    "rau xào đậu que":      "Rau xào đậu que",
    "rau xào đậu đũa":      "Rau xào đậu đũa",
    "trứng chiên":          "Trứng chiên",
    "trứng chiên thịt":     "Trứng chiên thịt",
    "không rõ":             "Không rõ",
}

EGG_SURCHARGE        = 6_000
YOLO_CONF_THRESHOLD  = 0.35
YOLO_EGG_CLASS_ID    = None   
YOLO_CLASS_NAMES     = {0: "egg half", 1: "egg whole"}

COMPARTMENTS = {
    "Top-Left":      (0.03, 0.03, 0.44, 0.48),
    "Top-Right":     (0.53, 0.03, 0.44, 0.48),
    "Bottom-Left":   (0.03, 0.55, 0.27, 0.42),
    "Bottom-Center": (0.34, 0.55, 0.32, 0.42),
    "Bottom-Right":  (0.70, 0.55, 0.27, 0.42),
}

def normalize_text(text):
    return unicodedata.normalize("NFC", text.strip().lower())

def load_class_names(path: str) -> list[str]:
    script_dir = pathlib.Path(__file__).parent
    for candidate in [pathlib.Path(path), script_dir / path]:
        if candidate.exists():
            names = [l.strip() for l in candidate.read_text(encoding="utf-8").splitlines() if l.strip()]
            return names
    return list(PRICE_MAP.keys())

@st.cache_resource(show_spinner="Đang tải mô hình CNN…")
def load_cnn():
    if not TF_AVAILABLE:
        return None
    try:
        return tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        st.warning(f"Không tải được CNN ({e}) — chạy ở **chế độ demo**.")
        return None

@st.cache_resource(show_spinner="Đang tải YOLO nhận diện trứng…")
def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        return _YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.warning(f"Không tải được YOLO ({e}) — đếm trứng ở **chế độ demo**.")
        return None

def crop_compartment(img_np: np.ndarray, region: tuple) -> np.ndarray:
    h, w = img_np.shape[:2]
    x  = int(region[0] * w);  y  = int(region[1] * h)
    cw = int(region[2] * w);  ch = int(region[3] * h)
    return img_np[y:y+ch, x:x+cw]

def predict_dish(model, crop_np: np.ndarray, class_names: list[str]) -> tuple[str, float]:
    if model is None:
        idx = np.random.randint(0, len(class_names))
        return class_names[idx], float(np.random.uniform(0.60, 0.97))
    resized = cv2.resize(crop_np, (IMG_SIZE, IMG_SIZE))
    inp     = np.expand_dims(resized.astype("float32") / 255.0, 0)
    preds   = model.predict(inp, verbose=0)[0]
    idx     = int(np.argmax(preds))
    name    = class_names[idx] if idx < len(class_names) else "không rõ"
    return name, float(preds[idx])

def count_eggs_yolo(yolo_model, img_np: np.ndarray) -> tuple[int, np.ndarray]:
    annotated = img_np.copy()

    if yolo_model is None:
        count = np.random.randint(1, 4)
        h, w  = img_np.shape[:2]
        for i in range(count):
            x1 = np.random.randint(0, w//2);  y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(60, 120)
            y2 = y1 + np.random.randint(60, 120)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255, 107, 53), 3)
            cv2.putText(annotated, f"egg {i+1}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,107,53), 2)
        return count, annotated

    results = yolo_model(img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    count   = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = YOLO_CLASS_NAMES.get(cls_id, f"egg cls{cls_id}")
        color = (255, 107, 53) if cls_id == 1 else (200, 80, 30)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return count, annotated


def fmt_vnd(amount: int) -> str:
    return f"{amount:,}₫".replace(",", ".")

st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: #0d1117;
  color: #c9d1d9;
}

/* ── Header ── */
.header-wrap {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.4rem 1.8rem;
  background: linear-gradient(110deg, #161b22 0%, #1a2233 100%);
  border: 1px solid #21262d;
  border-radius: 12px;
  margin-bottom: 1.6rem;
  position: relative;
  overflow: hidden;
}
.header-wrap::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px; height: 100%;
  background: linear-gradient(180deg, #ff6b35, #00d4aa);
  border-radius: 12px 0 0 12px;
}
.header-icon {
  font-size: 2.6rem;
  line-height: 1;
  filter: drop-shadow(0 0 12px rgba(255,107,53,0.5));
}
.header-title {
  font-family: 'Nunito', sans-serif;
  font-size: 1.7rem;
  font-weight: 900;
  color: #e8eaed;
  margin: 0;
  letter-spacing: -0.3px;
}
.header-title span { color: #ff6b35; }
.header-sub {
  font-size: 0.78rem;
  color: #6e7681;
  margin: 2px 0 0;
  font-weight: 400;
}
.demo-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(255,107,53,0.15);
  color: #ff6b35;
  border: 1px solid rgba(255,107,53,0.35);
  font-size: 0.65rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  padding: 3px 9px;
  border-radius: 20px;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin-left: 6px;
  vertical-align: middle;
}

/* ── Section headings ── */
.section-heading {
  font-family: 'Nunito', sans-serif;
  font-size: 0.7rem;
  font-weight: 800;
  color: #6e7681;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin: 1.4rem 0 0.7rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-heading::after {
  content: '';
  flex: 1;
  height: 1px;
  background: #21262d;
}

/* ── Compartment card ── */
.comp-card {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 10px;
  padding: 0.75rem;
  transition: border-color 0.2s, box-shadow 0.2s;
  height: 100%;
}
.comp-card:hover {
  border-color: rgba(255,107,53,0.4);
  box-shadow: 0 0 16px rgba(255,107,53,0.08);
}
.comp-slot-label {
  font-size: 0.62rem;
  font-weight: 700;
  color: #484f58;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.comp-dish-name {
  font-family: 'Nunito', sans-serif;
  font-size: 0.9rem;
  font-weight: 800;
  color: #e8eaed;
  margin: 0.45rem 0 0.35rem;
  line-height: 1.3;
}
.conf-track {
  background: #21262d;
  border-radius: 3px;
  height: 4px;
  margin: 0.2rem 0 0.3rem;
  overflow: hidden;
}
.conf-thumb {
  height: 4px;
  border-radius: 3px;
  background: linear-gradient(90deg, #ff6b35, #ffa07a);
}
.conf-label {
  font-size: 0.65rem;
  color: #484f58;
  margin-bottom: 0.35rem;
}
.price-chip {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  font-weight: 700;
  color: #00d4aa;
  background: rgba(0,212,170,0.08);
  border: 1px solid rgba(0,212,170,0.2);
  padding: 2px 8px;
  border-radius: 5px;
  margin-top: 0.1rem;
}
.price-chip.zero {
  color: #484f58;
  background: transparent;
  border-color: #21262d;
}

/* ── Egg detection panel ── */
.egg-panel {
  background: #161b22;
  border: 1px solid rgba(255,107,53,0.3);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.egg-panel::after {
  content: '🥚';
  position: absolute;
  right: 1rem; top: 50%;
  transform: translateY(-50%);
  font-size: 3rem;
  opacity: 0.08;
  pointer-events: none;
}
.egg-big {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.4rem;
  font-weight: 700;
  color: #ff6b35;
  line-height: 1;
}
.egg-sublabel {
  font-size: 0.72rem;
  color: #6e7681;
  margin-top: 2px;
}
.egg-note {
  font-size: 0.78rem;
  color: #6e7681;
  margin-top: 0.6rem;
}
.egg-surcharge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: #ff6b35;
  font-weight: 700;
  margin-top: 0.3rem;
}

/* ── Bill panel ── */
.bill-wrap {
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 10px;
  overflow: hidden;
}
.bill-header {
  background: linear-gradient(90deg, #1a2233, #161b22);
  padding: 0.7rem 1.1rem;
  border-bottom: 1px solid #21262d;
  display: flex;
  align-items: center;
  gap: 6px;
}
.bill-header-text {
  font-family: 'Nunito', sans-serif;
  font-size: 0.72rem;
  font-weight: 800;
  color: #6e7681;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.bill-body { padding: 0.2rem 0; }
.bill-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.42rem 1.1rem;
  border-bottom: 1px solid #0d1117;
  font-size: 0.83rem;
}
.bill-row:last-child { border-bottom: none; }
.bill-row-name { color: #c9d1d9; }
.bill-row-slot {
  font-size: 0.68rem;
  color: #484f58;
  margin-left: 4px;
}
.bill-row-price {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #8b949e;
  font-weight: 500;
}
.bill-egg-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.42rem 1.1rem;
  background: rgba(255,107,53,0.05);
  border-top: 1px dashed rgba(255,107,53,0.25);
  border-bottom: 1px dashed rgba(255,107,53,0.25);
  font-size: 0.83rem;
  color: #ff6b35;
}
.bill-egg-price {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: #ff6b35;
  font-weight: 700;
}
.bill-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, #30363d, transparent);
  margin: 0.1rem 1.1rem;
}
.bill-total-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem 1.1rem 0.9rem;
  background: rgba(0,212,170,0.04);
  border-top: 1px solid rgba(0,212,170,0.15);
}
.bill-total-label {
  font-family: 'Nunito', sans-serif;
  font-size: 0.75rem;
  font-weight: 800;
  color: #6e7681;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.bill-total-amount {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.25rem;
  font-weight: 700;
  color: #00d4aa;
}

/* ── Upload zone / info banner ── */
.info-banner {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  background: #161b22;
  border: 1px dashed #30363d;
  border-radius: 10px;
  padding: 1.2rem 1.4rem;
  color: #6e7681;
  font-size: 0.83rem;
}
.info-banner-icon { font-size: 1.5rem; opacity: 0.6; }

/* ── Streamlit overrides ── */
.stButton > button {
  background: linear-gradient(135deg, #ff6b35, #e85d27) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Nunito', sans-serif !important;
  font-weight: 800 !important;
  font-size: 0.9rem !important;
  padding: 0.55rem 1.4rem !important;
  transition: opacity 0.2s, transform 0.1s !important;
  box-shadow: 0 4px 14px rgba(255,107,53,0.3) !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton > button:disabled { opacity: 0.35 !important; transform: none !important; box-shadow: none !important; }

.stRadio > div { gap: 0.4rem !important; }
.stRadio label {
  font-size: 0.83rem !important;
  color: #8b949e !important;
}

[data-testid="stFileUploader"] {
  border: 1px dashed #30363d !important;
  border-radius: 10px !important;
  background: #161b22 !important;
}

div[data-testid="stImage"] img { border-radius: 8px; }

.stAlert {
  border-radius: 8px !important;
  font-size: 0.83rem !important;
}

hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()

demo = (cnn_model is None) or (yolo_model is None)
demo_chip = '<span class="demo-chip">⚡ Demo</span>' if demo else ""

st.markdown(f"""
<div class="header-wrap">
  <div class="header-icon">🍱</div>
  <div>
    <div class="header-title">Canteen <span>Auto-Billing</span>{demo_chip}</div>
    <div class="header-sub">Chụp khay → CNN nhận diện 5 ô → YOLO đếm trứng → In hóa đơn tức thì</div>
  </div>
</div>
""", unsafe_allow_html=True)

egg_price = EGG_SURCHARGE

col_in, _ = st.columns([1, 2])
with col_in:
    mode = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Chụp webcam"], horizontal=True)
    tray_image = None
    if mode == "Tải ảnh lên":
        up = st.file_uploader("Chọn ảnh khay cơm", type=["jpg","jpeg","png"])
        if up:
            tray_image = Image.open(up).convert("RGB")
    else:
        cam = st.camera_input("Hướng camera vào khay")
        if cam:
            tray_image = Image.open(cam).convert("RGB")

    go = st.button("🔍 Nhận diện món & tính tiền", disabled=(tray_image is None))

if tray_image and not go:
    st.image(tray_image, caption="Ảnh khay — sẵn sàng phân tích", width=500)

if go and tray_image:
    img_np = np.array(tray_image)

    cnn_results = {}
    with st.spinner("Đang phân loại từng ô bằng CNN…"):
        for slot, region in COMPARTMENTS.items():
            crop = crop_compartment(img_np, region)
            dish, conf = predict_dish(cnn_model, crop, CLASS_NAMES)
            dish_key = normalize_text(dish)
            price = PRICE_MAP.get(dish_key, 0)
            display = DISPLAY_NAMES.get(dish_key, dish)
            cnn_results[slot] = dict(crop=crop, dish=dish, display=display, conf=conf, price=price)
        
    has_thit_kho_trung = any(
        normalize_text(r["dish"]) == "thịt kho trứng" for r in cnn_results.values()
    )

    egg_count, annotated_np = 0, img_np.copy()
    if has_thit_kho_trung:
        with st.spinner("Đang đếm trứng bằng YOLO…"):
            egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    base_eggs   = 1 if has_thit_kho_trung else 0
    extra_eggs  = max(0, egg_count - base_eggs)
    egg_charge  = extra_eggs * egg_price

    st.markdown("<hr>", unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="section-heading">Các ô trong khay</div>', unsafe_allow_html=True)

        def render_card(col, slot, r):
            with col:
                pct = int(r["conf"] * 100)
                price_str = fmt_vnd(r["price"]) if r["price"] else "—"
                price_cls = "price-chip" if r["price"] else "price-chip zero"
                st.markdown(f'<div class="comp-card">', unsafe_allow_html=True)
                st.image(Image.fromarray(r["crop"]), use_container_width=True)
                st.markdown(f"""
                  <div class="comp-slot-label">{slot}</div>
                  <div class="comp-dish-name">{r["display"]}</div>
                  <div class="conf-track"><div class="conf-thumb" style="width:{pct}%"></div></div>
                  <div class="conf-label">{pct}% độ chính xác</div>
                  <div class="{price_cls}">{price_str}</div>
                </div>
                """, unsafe_allow_html=True)

        r1 = st.columns(2)
        for col, slot in zip(r1, ["Top-Left", "Top-Right"]):
            render_card(col, slot, cnn_results[slot])

        r2 = st.columns(3)
        for col, slot in zip(r2, ["Bottom-Left", "Bottom-Center", "Bottom-Right"]):
            render_card(col, slot, cnn_results[slot])
     
        if has_thit_kho_trung:
            st.markdown('<div class="section-heading">Nhận diện trứng</div>', unsafe_allow_html=True)
            st.image(annotated_np, caption=f"YOLO phát hiện {egg_count} quả trứng", use_container_width=True)

    with right_col:
        if has_thit_kho_trung:
            if extra_eggs > 0:
                surcharge_note = f"+{extra_eggs} trứng thêm × {fmt_vnd(egg_price)}"
                surcharge_html = f'<div class="egg-surcharge">+{fmt_vnd(egg_charge)} phụ thu</div>'
            else:
                surcharge_note = "1 trứng đã gồm trong Thịt kho trứng"
                surcharge_html = ""

            st.markdown(f"""
            <div class="section-heading">Trứng</div>
            <div class="egg-panel">
              <div class="egg-big">{egg_count}</div>
              <div class="egg-sublabel">quả trứng được phát hiện</div>
              <div class="egg-note">{surcharge_note}</div>
              {surcharge_html}
            </div>
            """, unsafe_allow_html=True)

        total = sum(r["price"] for r in cnn_results.values()) + egg_charge

        rows_html = ""
        for slot, r in cnn_results.items():
            price_str = fmt_vnd(r["price"]) if r["price"] > 0 else "&#8212;"
            rows_html += f"""
            <div class="bill-row">
              <span class="bill-row-name">{r["display"]} <span class="bill-row-slot">{slot}</span></span>
              <span class="bill-row-price">{price_str}</span>
            </div>"""

        egg_row_html = ""
        if egg_charge > 0:
            egg_row_html = f"""
            <div class="bill-egg-row">
              <span>🥚 Trứng thêm ×{extra_eggs}</span>
              <span class="bill-egg-price">+{fmt_vnd(egg_charge)}</span>
            </div>"""

        st.markdown(f"""
        <div class="section-heading">Hóa đơn</div>
        <div class="bill-wrap">
          <div class="bill-header">
            <span>🧾</span>
            <span class="bill-header-text">Chi tiết thanh toán</span>
          </div>
          <div class="bill-body">
            {rows_html}
            {egg_row_html}
            <div class="bill-divider"></div>
          </div>
          <div class="bill-total-row">
            <span class="bill-total-label">Tổng cộng</span>
            <span class="bill-total-amount">{fmt_vnd(total)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"✅ Tổng tiền: **{fmt_vnd(total)}**")

elif not tray_image:
    st.markdown("""
    <div class="info-banner">
      <span class="info-banner-icon">⬆️</span>
      <span>Tải ảnh khay cơm lên hoặc dùng webcam để bắt đầu nhận diện.</span>
    </div>
    """, unsafe_allow_html=True)