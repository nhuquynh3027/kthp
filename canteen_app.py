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

@st.cache_resource(show_spinner="Loading CNN model…")
def load_cnn():
    if not TF_AVAILABLE:
        return None
    try:
        return tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        st.warning(f"CNN model not loaded ({e}) — running in **demo mode**.")
        return None

@st.cache_resource(show_spinner="Loading YOLO egg detector…")
def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        return _YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.warning(f"YOLO model not loaded ({e}) — egg counting in **demo mode**.")
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
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Be Vietnam Pro', sans-serif;
  background-color: #F5F3EF;
  color: #1C1917;
}

/* ── Title bar ── */
.title-bar {
  background: #FFFFFF;
  border: 1px solid #E5E1D8;
  border-top: 3px solid #B45309;
  padding: 1rem 1.5rem;
  border-radius: 6px;
  margin-bottom: 1.4rem;
}
.title-bar h1 {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  color: #1C1917;
  letter-spacing: -0.3px;
}
.title-bar p {
  margin: .25rem 0 0;
  font-size: .8rem;
  color: #78716C;
  font-weight: 400;
}

/* ── Compartment card labels ── */
.crop-label {
  font-size: .65rem;
  color: #A8A29E;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-bottom: .25rem;
  font-weight: 500;
}
.dish-name {
  font-size: .92rem;
  font-weight: 700;
  color: #1C1917;
  margin: .3rem 0 .1rem;
}
.conf-bar {
  background: #E5E1D8;
  border-radius: 3px;
  height: 4px;
  margin: .25rem 0;
}
.conf-fill {
  background: #B45309;
  height: 4px;
  border-radius: 3px;
}
.price-tag {
  font-family: 'Space Mono', monospace;
  font-size: .85rem;
  color: #15803D;
  font-weight: 700;
}

/* ── Egg panel ── */
.egg-panel {
  background: #FFFBF5;
  border: 1px solid #FDE68A;
  border-left: 3px solid #D97706;
  border-radius: 6px;
  padding: 1rem 1.4rem;
  margin-bottom: 1rem;
}
.egg-count {
  font-size: 2.4rem;
  font-weight: 700;
  color: #B45309;
  font-family: 'Space Mono', monospace;
  line-height: 1;
}
.egg-label {
  font-size: .75rem;
  color: #92400E;
  margin-top: .1rem;
  font-weight: 500;
}

/* ── Bill panel ── */
.bill-panel {
  background: #FFFFFF;
  border: 1px solid #E5E1D8;
  border-radius: 6px;
  padding: 1.1rem 1.4rem;
}
.bill-row {
  display: flex;
  justify-content: space-between;
  padding: .32rem 0;
  border-bottom: 1px solid #F0EDE7;
  font-size: .85rem;
  color: #44403C;
}
.bill-egg {
  display: flex;
  justify-content: space-between;
  padding: .32rem 0;
  border-bottom: 1px dashed #FCD34D;
  font-size: .85rem;
  color: #B45309;
  font-weight: 600;
}
.bill-total {
  display: flex;
  justify-content: space-between;
  padding: .6rem 0 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: #1C1917;
  border-top: 2px solid #1C1917;
  margin-top: .4rem;
}
.mono {
  font-family: 'Space Mono', monospace;
  font-size: .82rem;
}
.mono-total {
  font-family: 'Space Mono', monospace;
  font-size: 1rem;
  color: #15803D;
}

/* ── Badge ── */
.badge {
  display: inline-block;
  background: #FEF3C7;
  color: #92400E;
  border: 1px solid #FCD34D;
  font-size: .62rem;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 10px;
  margin-left: 8px;
  vertical-align: middle;
  letter-spacing: .05em;
}

/* ── Streamlit overrides ── */
.stButton > button {
  background: #1C1917 !important;
  color: #F5F3EF !important;
  border: none !important;
  border-radius: 5px !important;
  font-family: 'Be Vietnam Pro', sans-serif !important;
  font-weight: 600 !important;
  font-size: .88rem !important;
  padding: .52rem 1.4rem !important;
  transition: background .15s !important;
}
.stButton > button:hover:not(:disabled) {
  background: #44403C !important;
}
.stButton > button:disabled {
  background: #D6D3CE !important;
  color: #A8A29E !important;
}

/* Container borders lighter */
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: #E5E1D8 !important;
  border-radius: 6px !important;
  background: #FFFFFF !important;
}

div[data-testid="stImage"] img { border-radius: 5px; }
.stAlert { border-radius: 5px !important; font-size: .83rem !important; }
hr { border: none !important; border-top: 1px solid #E5E1D8 !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()

demo = (cnn_model is None) or (yolo_model is None)
badge = '<span class="badge">DEMO</span>' if demo else ""

st.markdown(f"""
<div class="title-bar">
  <h1>🍱 Canteen Auto-Billing {badge}</h1>
  <p>Camera captures tray → CNN identifies 5 dishes → YOLO counts eggs → instant bill</p>
</div>
""", unsafe_allow_html=True)

egg_price = EGG_SURCHARGE

col_in, _ = st.columns([1, 2])
with col_in:
    mode = st.radio("Input source", ["Upload image", "Webcam snapshot"], horizontal=True)
    tray_image = None
    if mode == "Upload image":
        up = st.file_uploader("Choose tray photo", type=["jpg","jpeg","png"])
        if up:
            tray_image = Image.open(up).convert("RGB")
    else:
        cam = st.camera_input("Point camera at the tray")
        if cam:
            tray_image = Image.open(cam).convert("RGB")

    go = st.button("🔍 Identify dishes & calculate bill", disabled=(tray_image is None))

if tray_image and not go:
    st.image(tray_image, caption="Tray photo — ready to analyse", width=500)

if go and tray_image:
    img_np = np.array(tray_image)

    cnn_results = {}
    with st.spinner("Classifying compartments with CNN…"):
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
        with st.spinner("Counting eggs with YOLO…"):
            egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    base_eggs   = 1 if has_thit_kho_trung else 0
    extra_eggs  = max(0, egg_count - base_eggs)
    egg_charge  = extra_eggs * egg_price

    st.markdown("---")
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("### Tray compartments")

        def render_card(col, slot, r):
            with col:
                with st.container(border=True):
                    st.markdown(f'<div class="crop-label">{slot}</div>', unsafe_allow_html=True)
                    st.image(Image.fromarray(r["crop"]), use_container_width=True)
                    pct = int(r["conf"]*100)
                    price_str = fmt_vnd(r["price"]) if r["price"] else "—"
                    st.markdown(f"""
                    <div class="dish-name">{r["display"]}</div>
                    <div class="conf-bar"><div class="conf-fill" style="width:{pct}%"></div></div>
                    <div style="font-size:.68rem;color:#A8A29E;margin-bottom:.25rem">{pct}% confidence</div>
                    <div class="price-tag">{price_str}</div>
                    """, unsafe_allow_html=True)

        r1 = st.columns(2)
        for col, slot in zip(r1, ["Top-Left", "Top-Right"]):
            render_card(col, slot, cnn_results[slot])

        r2 = st.columns(3)
        for col, slot in zip(r2, ["Bottom-Left", "Bottom-Center", "Bottom-Right"]):
            render_card(col, slot, cnn_results[slot])
     
        if has_thit_kho_trung:
            st.markdown("### 🥚 Egg detection")
            st.image(annotated_np, caption=f"YOLO detected {egg_count} egg(s)", use_container_width=True)

    with right_col:
        if has_thit_kho_trung:
            surcharge_note = (
                f"+{extra_eggs} extra × {fmt_vnd(egg_price)}" if extra_eggs > 0
                else "1 egg included in Thịt kho trứng"
            )
            st.markdown(f"""
            <div class="egg-panel">
              <div class="egg-count">🥚 {egg_count}</div>
              <div class="egg-label">eggs detected by YOLO</div>
              <div style="margin-top:.5rem;font-size:.78rem;color:#92400E">{surcharge_note}</div>
              {'<div style="color:#B45309;font-weight:700;margin-top:.3rem;font-family:Space Mono,monospace;font-size:.85rem">+'
               + fmt_vnd(egg_charge) + ' surcharge</div>' if egg_charge > 0 else ""}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🧾 Bill")
        total = 0
        rows_html = ""
        for slot, r in cnn_results.items():
            price_str = fmt_vnd(r["price"]) if r["price"] > 0 else "&#8212;"
            rows_html += f"""<div class="bill-row">
              <span>{r["display"]} <span style="color:#A8A29E;font-size:.73rem">({slot})</span></span>
              <span class="mono">{price_str}</span></div>"""
            total += r["price"]

        egg_row_html = ""
        if egg_charge > 0:
            egg_row_html = f"""<div class="bill-egg">
              <span>🥚 Extra eggs ×{extra_eggs}</span>
              <span class="mono">+{fmt_vnd(egg_charge)}</span></div>"""
            total += egg_charge

        st.markdown(f"""
        <div class="bill-panel">
          {rows_html}
          {egg_row_html}
          <div class="bill-total">
            <span>TOTAL</span>
            <span class="mono-total">{fmt_vnd(total)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"✅ **{fmt_vnd(total)}**")

elif not tray_image:
    st.info("⬆️ Upload a tray photo or use the webcam to get started.")