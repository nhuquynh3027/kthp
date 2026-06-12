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
        st.warning(f"Không tải được CNN ({e}) — chạy ở chế độ demo.")
        return None

@st.cache_resource(show_spinner="Đang tải YOLO nhận diện trứng…")
def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        return _YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.warning(f"Không tải được YOLO ({e}) — đếm trứng ở chế độ demo.")
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
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (200, 80, 10), 3)
            cv2.putText(annotated, f"egg {i+1}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,80,10), 2)
        return count, annotated
    results = yolo_model(img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    count   = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = YOLO_CLASS_NAMES.get(cls_id, f"egg cls{cls_id}")
        color = (200, 80, 10) if cls_id == 1 else (160, 60, 0)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return count, annotated

def fmt_vnd(amount: int) -> str:
    return f"{amount:,}₫".replace(",", ".")

st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: #F7F5F2;
  color: #1A1612;
}

.hdr {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.8rem 1.4rem;
  padding: 1.8rem 0 1.2rem;
  border-bottom: 2px solid #1A1612;
  margin-bottom: 1.8rem;
}
.hdr-title {
  font-family: 'DM Serif Display', serif;
  font-size: 1.9rem;
  font-weight: 400;
  color: #1A1612;
  margin: 0;
  letter-spacing: -0.4px;
  line-height: 1;
}
.hdr-sub {
  font-size: 0.76rem;
  color: #8A7F74;
  font-weight: 400;
}
.demo-tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  font-weight: 600;
  color: #C8500A;
  border: 1px solid #C8500A;
  padding: 2px 6px;
  border-radius: 2px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  vertical-align: middle;
  margin-left: 0.6rem;
}

.sec-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  font-weight: 500;
  color: #8A7F74;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin: 1.6rem 0 0.75rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid #E4E0DA;
}

.ccard {
  background: #FFFFFF;
  border: 1px solid #E4E0DA;
  border-radius: 4px;
  padding: 0.75rem 0.8rem 0.85rem;
}
.ccard-slot {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.56rem;
  color: #B5ADA4;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.55rem;
}
.ccard-name {
  font-family: 'DM Serif Display', serif;
  font-size: 0.95rem;
  color: #1A1612;
  margin: 0.5rem 0 0.25rem;
  line-height: 1.2;
}
.conf-rail {
  background: #F0EDE8;
  border-radius: 1px;
  height: 2px;
  margin: 0.3rem 0 0.2rem;
}
.conf-bar-fill {
  height: 2px;
  border-radius: 1px;
  background: #C8500A;
}
.conf-pct {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.58rem;
  color: #B5ADA4;
}
.ccard-price {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.85rem;
  font-weight: 600;
  color: #1A6B4A;
  margin-top: 0.45rem;
  display: block;
}
.ccard-price.zero {
  color: #B5ADA4;
  font-weight: 400;
}

.egg-box {
  background: #FFF8F4;
  border: 1px solid #E8C9B4;
  border-radius: 4px;
  padding: 1rem 1.2rem;
  margin-bottom: 1.2rem;
}
.egg-num {
  font-family: 'DM Serif Display', serif;
  font-size: 2.8rem;
  color: #C8500A;
  line-height: 1;
}
.egg-unit {
  font-size: 0.71rem;
  color: #8A7F74;
  margin-top: 0.15rem;
}
.egg-note {
  font-size: 0.75rem;
  color: #8A7F74;
  margin-top: 0.6rem;
}
.egg-extra-cost {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  color: #C8500A;
  margin-top: 0.3rem;
}

.bill {
  background: #FFFFFF;
  border: 1px solid #E4E0DA;
  border-radius: 4px;
  overflow: hidden;
}
.bill-top {
  padding: 0.65rem 1.1rem 0.55rem;
  border-bottom: 1px solid #E4E0DA;
}
.bill-top-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: #8A7F74;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.b-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.36rem 1.1rem;
  font-size: 0.82rem;
  color: #3A332C;
  border-bottom: 1px solid #F7F5F2;
}
.b-slot {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.58rem;
  color: #B5ADA4;
  margin-left: 5px;
}
.b-amount {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  color: #6B6259;
}
.b-egg-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.36rem 1.1rem;
  font-size: 0.82rem;
  color: #C8500A;
  border-top: 1px dashed #E8C9B4;
  border-bottom: 1px dashed #E8C9B4;
  background: #FFF8F4;
}
.b-egg-amount {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  color: #C8500A;
}
.b-total-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.8rem 1.1rem;
  border-top: 2px solid #1A1612;
}
.b-total-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #1A1612;
}
.b-total-amount {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.3rem;
  font-weight: 600;
  color: #1A6B4A;
}

.empty-state {
  background: #FFFFFF;
  border: 1px dashed #C8C0B6;
  border-radius: 4px;
  padding: 2.4rem 1.5rem;
  text-align: center;
  color: #8A7F74;
  font-size: 0.82rem;
}

.stButton > button {
  background: #1A1612 !important;
  color: #F7F5F2 !important;
  border: none !important;
  border-radius: 3px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.84rem !important;
  padding: 0.52rem 1.5rem !important;
  letter-spacing: 0.01em !important;
  transition: background 0.15s !important;
  box-shadow: none !important;
}
.stButton > button:hover:not(:disabled) {
  background: #3A332C !important;
}
.stButton > button:disabled {
  background: #C8C0B6 !important;
  color: #F7F5F2 !important;
}

.stRadio label { font-size: 0.82rem !important; color: #3A332C !important; }
[data-testid="stFileUploader"] {
  border: 1px dashed #C8C0B6 !important;
  border-radius: 4px !important;
  background: #FFFFFF !important;
}
div[data-testid="stImage"] img { border-radius: 3px; }
.stAlert { border-radius: 3px !important; font-size: 0.81rem !important; }
hr { border: none !important; border-top: 1px solid #E4E0DA !important; }
[data-testid="column"] { padding: 0 0.35rem !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()

demo = (cnn_model is None) or (yolo_model is None)
demo_tag = '<span class="demo-tag">Demo</span>' if demo else ""

st.markdown(f"""
<div class="hdr">
  <span class="hdr-title">Canteen Auto-Billing{demo_tag}</span>
  <span class="hdr-sub">Ảnh khay &rarr; CNN nhận diện 5 ô &rarr; YOLO đếm trứng &rarr; Hóa đơn tức thì</span>
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

    go = st.button("Nhận diện món & tính tiền", disabled=(tray_image is None))

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
        st.markdown('<div class="sec-label">Các ô trong khay</div>', unsafe_allow_html=True)

        def render_card(col, slot, r):
            with col:
                pct = int(r["conf"] * 100)
                price_str = fmt_vnd(r["price"]) if r["price"] else "—"
                price_cls = "ccard-price" if r["price"] else "ccard-price zero"
                st.markdown('<div class="ccard">', unsafe_allow_html=True)
                st.image(Image.fromarray(r["crop"]), use_container_width=True)
                st.markdown(f"""
                  <div class="ccard-slot">{slot}</div>
                  <div class="ccard-name">{r["display"]}</div>
                  <div class="conf-rail"><div class="conf-bar-fill" style="width:{pct}%"></div></div>
                  <div class="conf-pct">{pct}%</div>
                  <span class="{price_cls}">{price_str}</span>
                </div>
                """, unsafe_allow_html=True)

        r1 = st.columns(2)
        for col, slot in zip(r1, ["Top-Left", "Top-Right"]):
            render_card(col, slot, cnn_results[slot])

        r2 = st.columns(3)
        for col, slot in zip(r2, ["Bottom-Left", "Bottom-Center", "Bottom-Right"]):
            render_card(col, slot, cnn_results[slot])

        if has_thit_kho_trung:
            st.markdown('<div class="sec-label">Kết quả nhận diện trứng</div>', unsafe_allow_html=True)
            st.image(annotated_np, caption=f"YOLO phát hiện {egg_count} quả trứng", use_container_width=True)

    with right_col:
        if has_thit_kho_trung:
            if extra_eggs > 0:
                surcharge_note = f"+{extra_eggs} quả thêm × {fmt_vnd(egg_price)}"
                extra_html = f'<div class="egg-extra-cost">+{fmt_vnd(egg_charge)} phụ thu</div>'
            else:
                surcharge_note = "1 trứng đã gồm trong Thịt kho trứng"
                extra_html = ""

            st.markdown(f"""
            <div class="sec-label">Trứng</div>
            <div class="egg-box">
              <div class="egg-num">{egg_count}</div>
              <div class="egg-unit">quả trứng được phát hiện</div>
              <div class="egg-note">{surcharge_note}</div>
              {extra_html}
            </div>
            """, unsafe_allow_html=True)

        total = sum(r["price"] for r in cnn_results.values()) + egg_charge

        rows_html = ""
        for slot, r in cnn_results.items():
            price_str = fmt_vnd(r["price"]) if r["price"] > 0 else "&#8212;"
            rows_html += f"""
            <div class="b-row">
              <span>{r["display"]}<span class="b-slot">{slot}</span></span>
              <span class="b-amount">{price_str}</span>
            </div>"""

        egg_row_html = ""
        if egg_charge > 0:
            egg_row_html = f"""
            <div class="b-egg-row">
              <span>Trứng thêm &times;{extra_eggs}</span>
              <span class="b-egg-amount">+{fmt_vnd(egg_charge)}</span>
            </div>"""

        st.markdown(f"""
        <div class="sec-label">Hóa đơn</div>
        <div class="bill">
          <div class="bill-top">
            <div class="bill-top-label">Chi tiết thanh toán</div>
          </div>
          <div>
            {rows_html}
            {egg_row_html}
          </div>
          <div class="b-total-row">
            <span class="b-total-label">Tổng cộng</span>
            <span class="b-total-amount">{fmt_vnd(total)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"Tổng tiền: **{fmt_vnd(total)}**")

elif not tray_image:
    st.markdown("""
    <div class="empty-state">
      Tải ảnh khay cơm lên hoặc dùng webcam để bắt đầu.
    </div>
    """, unsafe_allow_html=True)