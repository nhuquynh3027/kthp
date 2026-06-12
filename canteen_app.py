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
        st.warning(f"Không tải được mô hình CNN ({e}) — chạy ở chế độ demo.")
        return None

@st.cache_resource(show_spinner="Đang tải YOLO egg detector…")
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
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255, 200, 0), 3)
            cv2.putText(annotated, f"egg {i+1}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)
        return count, annotated

    results = yolo_model(img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    count   = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = YOLO_CLASS_NAMES.get(cls_id, f"egg cls{cls_id}")
        color = (255, 200, 0) if cls_id == 1 else (200, 150, 0)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return count, annotated


def fmt_vnd(amount: int) -> str:
    return f"{amount:,}₫".replace(",", ".")

st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background: #FAFAF8;
  color: #18181B;
}

/* ── Header ── */
.app-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 1.2rem 1.6rem;
  background: #18181B;
  border-radius: 12px;
  margin-bottom: 1.5rem;
}
.app-header-icon {
  font-size: 2rem;
  line-height: 1;
}
.app-header-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: #FAFAF8;
  margin: 0;
  letter-spacing: -0.2px;
}
.app-header-sub {
  font-size: 0.75rem;
  color: #71717A;
  margin: 2px 0 0;
}
.demo-pill {
  margin-left: auto;
  background: #EF4444;
  color: #fff;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 100px;
}

/* ── Section label ── */
.section-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #A1A1AA;
  margin-bottom: 0.6rem;
}

/* ── Dish card ── */
.dish-card {
  background: #fff;
  border: 1px solid #E4E4E7;
  border-radius: 10px;
  padding: 0.75rem;
  height: 100%;
}
.dish-slot {
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #A1A1AA;
  margin-bottom: 0.5rem;
}
.dish-name {
  font-size: 0.9rem;
  font-weight: 600;
  color: #18181B;
  margin: 0.5rem 0 0.2rem;
  line-height: 1.3;
}
.conf-track {
  background: #F4F4F5;
  height: 3px;
  border-radius: 2px;
  margin: 0.35rem 0 0.2rem;
  overflow: hidden;
}
.conf-fill {
  height: 3px;
  border-radius: 2px;
  background: #18181B;
}
.conf-text {
  font-size: 0.65rem;
  color: #A1A1AA;
  margin-bottom: 0.4rem;
}
.dish-price {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.88rem;
  font-weight: 500;
  color: #16A34A;
}

/* ── Egg panel ── */
.egg-panel {
  background: #FFFBEB;
  border: 1px solid #FDE68A;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.egg-big {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.2rem;
  font-weight: 600;
  color: #92400E;
  line-height: 1;
}
.egg-meta-label {
  font-size: 0.7rem;
  color: #92400E;
  font-weight: 500;
  margin-bottom: 2px;
}
.egg-meta-note {
  font-size: 0.72rem;
  color: #B45309;
}
.egg-surcharge {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  font-weight: 500;
  color: #B45309;
  background: #FEF3C7;
  border: 1px solid #FDE68A;
  border-radius: 6px;
  padding: 4px 10px;
}

/* ── Bill card ── */
.bill-card {
  background: #fff;
  border: 1px solid #E4E4E7;
  border-radius: 10px;
  overflow: hidden;
}
.bill-header {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #F4F4F5;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #71717A;
}
.bill-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 1rem;
  border-bottom: 1px solid #F4F4F5;
  font-size: 0.83rem;
  color: #3F3F46;
}
.bill-row-slot {
  font-size: 0.65rem;
  color: #A1A1AA;
  margin-left: 5px;
}
.bill-row-egg {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 1rem;
  border-bottom: 1px solid #FDE68A;
  font-size: 0.83rem;
  color: #92400E;
  font-weight: 500;
  background: #FFFBEB;
}
.bill-amount {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.8rem;
  color: #3F3F46;
}
.bill-total-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.85rem 1rem;
  background: #18181B;
}
.bill-total-label {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #A1A1AA;
}
.bill-total-amount {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.05rem;
  font-weight: 600;
  color: #4ADE80;
}

/* ── Streamlit overrides ── */
.stButton > button {
  background: #18181B !important;
  color: #FAFAF8 !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  padding: 0.55rem 1.4rem !important;
  transition: background 0.15s !important;
}
.stButton > button:hover:not(:disabled) {
  background: #3F3F46 !important;
}
.stButton > button:disabled {
  background: #E4E4E7 !important;
  color: #A1A1AA !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: #E4E4E7 !important;
  border-radius: 10px !important;
  background: #fff !important;
}
div[data-testid="stImage"] img { border-radius: 8px; }
.stAlert { border-radius: 8px !important; font-size: 0.82rem !important; }
hr { border: none !important; border-top: 1px solid #E4E4E7 !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()

demo = (cnn_model is None) or (yolo_model is None)
demo_html = '<span class="demo-pill">DEMO</span>' if demo else ""

st.markdown(f"""
<div class="app-header">
  <div class="app-header-icon">🍱</div>
  <div>
    <div class="app-header-title">Canteen Auto-Billing</div>
    <div class="app-header-sub">Chụp khay → CNN nhận diện 5 món → YOLO đếm trứng → tính tiền tự động</div>
  </div>
  {demo_html}
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
    with st.spinner("Đang phân tích từng ô với CNN…"):
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
        with st.spinner("Đang đếm trứng với YOLO…"):
            egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    base_eggs   = 1 if has_thit_kho_trung else 0
    extra_eggs  = max(0, egg_count - base_eggs)
    egg_charge  = extra_eggs * egg_price

    st.markdown("---")
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown('<div class="section-label">Các ô trong khay</div>', unsafe_allow_html=True)

        def render_card(col, slot, r):
            with col:
                pct = int(r["conf"] * 100)
                price_str = fmt_vnd(r["price"]) if r["price"] else "—"
                bar_color = "#16A34A" if pct >= 80 else "#EAB308" if pct >= 60 else "#EF4444"
                st.image(Image.fromarray(r["crop"]), use_container_width=True)
                st.markdown(f"""
                <div style="padding: 0.1rem 0 0.6rem;">
                  <div class="dish-slot">{slot}</div>
                  <div class="dish-name">{r["display"]}</div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{pct}%; background:{bar_color};"></div>
                  </div>
                  <div class="conf-text">{pct}% tin cậy</div>
                  <div class="dish-price">{price_str}</div>
                </div>
                """, unsafe_allow_html=True)

        r1 = st.columns(2)
        for col, slot in zip(r1, ["Top-Left", "Top-Right"]):
            render_card(col, slot, cnn_results[slot])

        r2 = st.columns(3)
        for col, slot in zip(r2, ["Bottom-Left", "Bottom-Center", "Bottom-Right"]):
            render_card(col, slot, cnn_results[slot])
     
        if has_thit_kho_trung:
            st.markdown('<div class="section-label" style="margin-top:1.2rem;">Phát hiện trứng (YOLO)</div>', unsafe_allow_html=True)
            st.image(annotated_np, caption=f"YOLO phát hiện {egg_count} quả trứng", use_container_width=True)

    with right_col:
        if has_thit_kho_trung:
            surcharge_note = (
                f"+{extra_eggs} quả thêm × {fmt_vnd(egg_price)}" if extra_eggs > 0
                else "1 trứng đã gồm trong thịt kho trứng"
            )
            egg_surcharge_html = ""
            if egg_charge > 0:
                egg_surcharge_html = f'<div class="egg-surcharge">+{fmt_vnd(egg_charge)}</div>'

            st.markdown(f"""
            <div class="egg-panel">
              <div>
                <div class="egg-big">🥚 {egg_count}</div>
              </div>
              <div>
                <div class="egg-meta-label">Trứng phát hiện được</div>
                <div class="egg-meta-note">{surcharge_note}</div>
              </div>
              {egg_surcharge_html}
            </div>
            """, unsafe_allow_html=True)

        # Bill
        total = 0
        rows_html = ""
        for slot, r in cnn_results.items():
            price_str = fmt_vnd(r["price"]) if r["price"] > 0 else "&#8212;"
            rows_html += f"""<div class="bill-row">
              <span>{r["display"]} <span class="bill-row-slot">{slot}</span></span>
              <span class="bill-amount">{price_str}</span>
            </div>"""
            total += r["price"]

        egg_row_html = ""
        if egg_charge > 0:
            egg_row_html = f"""<div class="bill-row-egg">
              <span>🥚 Trứng thêm ×{extra_eggs}</span>
              <span class="bill-amount" style="color:#92400E;">+{fmt_vnd(egg_charge)}</span>
            </div>"""
            total += egg_charge

        st.markdown(f"""
        <div class="bill-card">
          <div class="bill-header">Chi tiết thanh toán</div>
          {rows_html}
          {egg_row_html}
          <div class="bill-total-row">
            <span class="bill-total-label">Tổng cộng</span>
            <span class="bill-total-amount">{fmt_vnd(total)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"✅ **{fmt_vnd(total)}**")

elif not tray_image:
    st.info("⬆️ Tải ảnh khay cơm hoặc dùng webcam để bắt đầu.")