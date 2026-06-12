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
        color = (255, 200, 0) if cls_id == 1 else (200, 150, 0)  # whole=bright, half=dimmer
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return count, annotated


def fmt_vnd(amount: int) -> str:
    return f"{amount:,}₫".replace(",", ".")

st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700&family=Space+Mono:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

.title-bar {
  background: linear-gradient(135deg,#1a1f2e,#16213e);
  border-left: 4px solid #f4a261;
  padding: 1rem 1.5rem; border-radius: 6px; margin-bottom: 1.2rem;
}
.title-bar h1 { margin:0; font-size:1.75rem; color:#f4a261; }
.title-bar p  { margin:.2rem 0 0; font-size:.82rem; color:#8899aa; }

.crop-label { font-size:.7rem; color:#8899aa; text-transform:uppercase;
              letter-spacing:1px; margin-bottom:.25rem; }
.dish-name  { font-size:.95rem; font-weight:700; color:#f4a261; margin:.3rem 0 .1rem; }
.conf-bar   { background:#2a3348; border-radius:4px; height:5px; margin:.25rem 0; }
.conf-fill  { background:#f4a261; height:5px; border-radius:4px; }
.price-tag  { font-family:'Space Mono',monospace; font-size:.88rem;
              color:#4ecdc4; font-weight:700; }

.egg-panel {
  background:#1a1f2e; border:1px solid #f4a261; border-radius:8px;
  padding:1rem 1.4rem; margin-bottom:1rem;
}
.egg-count { font-size:2.2rem; font-weight:700; color:#f4a261;
             font-family:'Space Mono',monospace; }
.egg-label { font-size:.8rem; color:#8899aa; margin-top:-.3rem; }

.bill-panel { background:#1a1f2e; border:1px solid #2a3348;
              border-radius:8px; padding:1.1rem 1.4rem; }
.bill-row   { display:flex; justify-content:space-between; padding:.3rem 0;
              border-bottom:1px solid #2a3348; font-size:.88rem; }
.bill-egg   { display:flex; justify-content:space-between; padding:.3rem 0;
              border-bottom:1px dashed #f4a261; font-size:.88rem; color:#f4a261; }
.bill-total { display:flex; justify-content:space-between; padding:.55rem 0 0;
              font-size:1.1rem; font-weight:700; color:#f4a261; }
.mono       { font-family:'Space Mono',monospace; }

.badge { display:inline-block; background:#f4a261; color:#0f1117;
         font-size:.68rem; font-weight:700; padding:2px 7px;
         border-radius:10px; margin-left:8px; vertical-align:middle; }
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
                    <div style="font-size:.7rem;color:#556677">{pct}% confidence</div>
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
              <div style="margin-top:.5rem;font-size:.82rem;color:#8899aa">{surcharge_note}</div>
              {'<div style="color:#f4a261;font-weight:700;margin-top:.3rem">+'
               + fmt_vnd(egg_charge) + ' surcharge</div>' if egg_charge > 0 else ""}
            </div>
            """, unsafe_allow_html=True)

      
        st.markdown("### 🧾 Bill")
        total = 0
        rows_html = ""
        for slot, r in cnn_results.items():
            price_str = fmt_vnd(r["price"]) if r["price"] > 0 else "&#8212;"
            rows_html += f"""<div class="bill-row">
              <span>{r["display"]} <span style="color:#556677;font-size:.78rem">({slot})</span></span>
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
            <span class="mono">{fmt_vnd(total)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"✅ **{fmt_vnd(total)}**")

elif not tray_image:
    st.info("⬆️ Upload a tray photo or use the webcam to get started.")
