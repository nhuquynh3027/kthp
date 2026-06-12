import os, pathlib
import unicodedata
import streamlit as st
import numpy as np
from PIL import Image
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
YOLO_CLASS_NAMES     = {0: "egg half", 1: "egg whole"}

COMPARTMENTS = {
    "Top-Left":      (0.03, 0.03, 0.44, 0.48),
    "Top-Right":     (0.53, 0.03, 0.44, 0.48),
    "Bottom-Left":   (0.03, 0.55, 0.27, 0.42),
    "Bottom-Center": (0.34, 0.55, 0.32, 0.42),
    "Bottom-Right":  (0.70, 0.55, 0.27, 0.42),
}

def normalize_text(t):
    return unicodedata.normalize("NFC", t.strip().lower())

def load_class_names(path):
    sd = pathlib.Path(__file__).parent
    for c in [pathlib.Path(path), sd / path]:
        if c.exists():
            return [l.strip() for l in c.read_text(encoding="utf-8").splitlines() if l.strip()]
    return list(PRICE_MAP.keys())

@st.cache_resource(show_spinner="Đang tải mô hình CNN…")
def load_cnn():
    if not TF_AVAILABLE: return None
    try: return tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        st.warning(f"CNN: {e}"); return None

@st.cache_resource(show_spinner="Đang tải YOLO…")
def load_yolo():
    if not YOLO_AVAILABLE: return None
    try: return _YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.warning(f"YOLO: {e}"); return None

def crop_compartment(img, region):
    h, w = img.shape[:2]
    x, y = int(region[0]*w), int(region[1]*h)
    return img[y:y+int(region[3]*h), x:x+int(region[2]*w)]

def predict_dish(model, crop, class_names):
    if model is None:
        idx = np.random.randint(0, len(class_names))
        return class_names[idx], float(np.random.uniform(0.65, 0.98))
    r = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    p = model.predict(np.expand_dims(r.astype("float32")/255., 0), verbose=0)[0]
    i = int(np.argmax(p))
    return (class_names[i] if i < len(class_names) else "không rõ"), float(p[i])

def count_eggs_yolo(yolo, img):
    ann = img.copy()
    if yolo is None:
        n = np.random.randint(1, 4); h, w = img.shape[:2]
        for i in range(n):
            x1,y1 = np.random.randint(0,w//2), np.random.randint(0,h//2)
            cv2.rectangle(ann,(x1,y1),(x1+100,y1+100),(255,200,0),3)
            cv2.putText(ann,f"egg {i+1}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,200,0),2)
        return n, ann
    res = yolo(img, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    n = 0
    for box in res.boxes:
        n += 1; x1,y1,x2,y2 = map(int,box.xyxy[0])
        lbl = YOLO_CLASS_NAMES.get(int(box.cls[0]),f"egg")
        cv2.rectangle(ann,(x1,y1),(x2,y2),(255,200,0),3)
        cv2.putText(ann,f"{lbl} {float(box.conf[0]):.0%}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,200,0),2)
    return n, ann

def fmt(n): return f"{n:,}₫".replace(",",".")

# ═══════════════════════════════════════════
st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Sora', sans-serif;
  background: #F2F0EB;
  color: #1C1B18;
}

/* ════ HERO HEADER ════ */
.hero {
  position: relative;
  background: #FFFDF7;
  border: 1px solid #E2DFD5;
  border-radius: 20px;
  padding: 28px 32px 24px;
  margin-bottom: 28px;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #E8622A 0%, #F0A835 50%, #E8622A 100%);
  border-radius: 20px 20px 0 0;
}
.hero-eyebrow {
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: #E8622A; margin-bottom: 8px;
}
.hero-title {
  font-size: 28px; font-weight: 700;
  color: #1C1B18; line-height: 1.15;
  letter-spacing: -0.5px; margin-bottom: 6px;
}
.hero-title span { color: #E8622A; }
.hero-sub {
  font-size: 13px; color: #8A887E; font-weight: 400; line-height: 1.5;
}
.hero-pills {
  display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap;
}
.hero-pill {
  display: inline-flex; align-items: center; gap: 5px;
  background: #F2F0EB; border: 1px solid #E2DFD5;
  border-radius: 100px; padding: 4px 12px;
  font-size: 12px; color: #5A5850; font-weight: 500;
}
.hero-pill-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #E8622A; flex-shrink: 0;
}
.demo-badge {
  position: absolute; top: 20px; right: 24px;
  background: #FEF3E8; color: #C04B10;
  border: 1px solid #FCDDB8;
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  padding: 5px 12px; border-radius: 100px;
}

/* ════ UPLOAD PANEL ════ */
.upload-label {
  font-size: 12px; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: #8A887E; margin-bottom: 8px;
}

/* ════ SECTION HEADER ════ */
.sec-head {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 16px; padding-bottom: 12px;
  border-bottom: 1px solid #E2DFD5;
}
.sec-num {
  width: 24px; height: 24px; border-radius: 6px;
  background: #E8622A; color: #fff;
  font-size: 12px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.sec-title { font-size: 15px; font-weight: 600; color: #1C1B18; }

/* ════ DISH CARD ════ */
.d-card {
  background: #FFFDF7;
  border: 1px solid #E2DFD5;
  border-radius: 14px;
  overflow: hidden;
}
.d-img { width:100%; aspect-ratio:1/1; object-fit:cover; display:block; }
.d-body { padding: 10px 12px 12px; }
.d-tag {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #B8B5AC; margin-bottom: 5px;
}
.d-name { font-size: 13px; font-weight: 600; color: #1C1B18; line-height: 1.3; margin-bottom: 7px; }
.d-bar-bg { background: #EAE8E2; height: 3px; border-radius: 2px; margin-bottom: 3px; }
.d-bar-fill { height: 3px; border-radius: 2px; }
.d-conf { font-size: 11px; color: #B8B5AC; margin-bottom: 7px; }
.d-price {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 14px; font-weight: 500; color: #27884A;
}

/* ════ EGG PANEL ════ */
.egg-box {
  background: #FFFDF7;
  border: 1px solid #E2DFD5;
  border-left: 3px solid #E8622A;
  border-radius: 0 12px 12px 0;
  padding: 16px 18px;
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 14px;
}
.egg-count {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 40px; font-weight: 500;
  color: #E8622A; line-height: 1; flex-shrink: 0;
}
.egg-label { font-size: 13px; font-weight: 600; color: #1C1B18; margin-bottom: 2px; }
.egg-note { font-size: 12px; color: #8A887E; }
.egg-charge {
  margin-left: auto; flex-shrink: 0;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px; font-weight: 500; color: #C04B10;
  background: #FEF3E8; border: 1px solid #FCDDB8;
  border-radius: 8px; padding: 6px 12px;
}

/* ════ BILL ════ */
.bill {
  background: #FFFDF7;
  border: 1px solid #E2DFD5;
  border-radius: 16px;
  overflow: hidden;
}
.bill-head {
  padding: 14px 18px;
  border-bottom: 1px solid #EAE8E2;
  display: flex; align-items: center; gap: 8px;
}
.bill-head-icon {
  width: 28px; height: 28px; border-radius: 8px;
  background: #1C1B18; color: #F2F0EB;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; flex-shrink: 0;
}
.bill-head-title { font-size: 14px; font-weight: 600; color: #1C1B18; }
.bill-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 10px 18px;
  border-bottom: 1px solid #F5F3EE;
  font-size: 13px; color: #3D3C38;
}
.bill-row-sub { font-size: 11px; color: #B8B5AC; margin-left: 5px; }
.bill-row-amt {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px; color: #5A5850; flex-shrink: 0; margin-left: 8px;
}
.bill-egg-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 10px 18px;
  border-bottom: 1px solid #FCDDB8;
  background: #FEF3E8;
  font-size: 13px; color: #C04B10; font-weight: 500;
}
.bill-footer {
  padding: 18px;
  background: #1C1B18;
  display: flex; justify-content: space-between; align-items: center;
}
.bill-footer-label {
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase; color: #5A5850;
}
.bill-footer-total {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 22px; font-weight: 500; color: #7BDE97;
}

/* ════ STREAMLIT OVERRIDES ════ */
.stButton > button {
  width: 100% !important;
  background: #E8622A !important; color: #fff !important;
  border: none !important; border-radius: 12px !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 14px !important; font-weight: 600 !important;
  padding: 12px 24px !important;
  letter-spacing: 0.01em !important;
  transition: background .15s, transform .1s !important;
}
.stButton > button:hover:not(:disabled) {
  background: #C94E1E !important; transform: translateY(-1px) !important;
}
.stButton > button:disabled {
  background: #E2DFD5 !important; color: #B8B5AC !important;
  transform: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: #E2DFD5 !important;
  border-radius: 14px !important;
  background: #FFFDF7 !important;
}
div[data-testid="stImage"] img { border-radius: 10px; }
.stAlert { border-radius: 12px !important; font-size: 13px !important; }
hr { border:none !important; border-top: 1px solid #E2DFD5 !important; }
.stRadio label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()
demo = (cnn_model is None) or (yolo_model is None)

# ─── HERO ───
demo_badge = '<div class="demo-badge">Demo mode</div>' if demo else ""
st.markdown(f"""
<div class="hero">
  {demo_badge}
  <div class="hero-eyebrow">AI-Powered · Canteen System</div>
  <div class="hero-title">Canteen<br><span>Auto-Billing</span></div>
  <div class="hero-sub">Chụp khay cơm — nhận diện món ăn và tính tiền tự động trong vài giây.</div>
  <div class="hero-pills">
    <span class="hero-pill"><span class="hero-pill-dot"></span>CNN nhận diện 5 ô</span>
    <span class="hero-pill"><span class="hero-pill-dot"></span>YOLO đếm trứng</span>
    <span class="hero-pill"><span class="hero-pill-dot"></span>Xuất hóa đơn tức thì</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── INPUT ───
col_in, _ = st.columns([1, 2])
with col_in:
    st.markdown('<div class="upload-label">Nguồn ảnh</div>', unsafe_allow_html=True)
    mode = st.radio("src", ["Tải ảnh lên", "Chụp webcam"], horizontal=True, label_visibility="collapsed")
    tray_image = None
    if mode == "Tải ảnh lên":
        up = st.file_uploader("img", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up: tray_image = Image.open(up).convert("RGB")
    else:
        cam = st.camera_input("cam", label_visibility="collapsed")
        if cam: tray_image = Image.open(cam).convert("RGB")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    go = st.button("🔍  Nhận diện & tính tiền", disabled=(tray_image is None))

if tray_image and not go:
    with col_in:
        st.image(tray_image, use_container_width=True)

# ─── RESULTS ───
if go and tray_image:
    img_np = np.array(tray_image)
    cnn_results = {}
    with st.spinner("Đang phân tích…"):
        for slot, region in COMPARTMENTS.items():
            crop = crop_compartment(img_np, region)
            dish, conf = predict_dish(cnn_model, crop, CLASS_NAMES)
            dk = normalize_text(dish)
            cnn_results[slot] = dict(
                crop=crop, dish=dish,
                display=DISPLAY_NAMES.get(dk, dish),
                conf=conf, price=PRICE_MAP.get(dk, 0)
            )

    has_egg_dish = any(normalize_text(r["dish"]) == "thịt kho trứng" for r in cnn_results.values())
    egg_count, annotated_np = 0, img_np.copy()
    if has_egg_dish:
        with st.spinner("Đang đếm trứng…"):
            egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    extra_eggs = max(0, egg_count - (1 if has_egg_dish else 0))
    egg_charge = extra_eggs * EGG_SURCHARGE

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("""
        <div class="sec-head">
          <div class="sec-num">1</div>
          <div class="sec-title">Các ô trong khay</div>
        </div>""", unsafe_allow_html=True)

        def dish_card(col, slot, r):
            pct = int(r["conf"]*100)
            bc  = "#27884A" if pct>=80 else "#C07A10" if pct>=60 else "#B03030"
            price_str = fmt(r["price"]) if r["price"] else "—"
            with col:
                st.image(Image.fromarray(r["crop"]), use_container_width=True)
                st.markdown(f"""
                <div style="padding:4px 0 12px">
                  <div class="d-tag">{slot}</div>
                  <div class="d-name">{r["display"]}</div>
                  <div class="d-bar-bg"><div class="d-bar-fill" style="width:{pct}%;background:{bc}"></div></div>
                  <div class="d-conf">{pct}% tin cậy</div>
                  <div class="d-price">{price_str}</div>
                </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="small")
        dish_card(c1, "Top-Left",  cnn_results["Top-Left"])
        dish_card(c2, "Top-Right", cnn_results["Top-Right"])
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3, gap="small")
        dish_card(c3, "Bottom-Left",   cnn_results["Bottom-Left"])
        dish_card(c4, "Bottom-Center", cnn_results["Bottom-Center"])
        dish_card(c5, "Bottom-Right",  cnn_results["Bottom-Right"])

        if has_egg_dish:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="sec-head">
              <div class="sec-num">2</div>
              <div class="sec-title">Phát hiện trứng — YOLO</div>
            </div>""", unsafe_allow_html=True)
            st.image(annotated_np, caption=f"Phát hiện {egg_count} quả trứng", use_container_width=True)

    with right:
        if has_egg_dish:
            note  = f"+{extra_eggs} quả thêm × {fmt(EGG_SURCHARGE)}" if extra_eggs > 0 else "1 trứng đã bao gồm trong giá"
            cpill = f'<div class="egg-charge">+{fmt(egg_charge)}</div>' if egg_charge > 0 else ""
            st.markdown(f"""
            <div class="egg-box">
              <div class="egg-count">{egg_count}</div>
              <div>
                <div class="egg-label">Trứng phát hiện</div>
                <div class="egg-note">{note}</div>
              </div>
              {cpill}
            </div>""", unsafe_allow_html=True)

        total = sum(r["price"] for r in cnn_results.values()) + egg_charge
        rows = "".join(f"""
        <div class="bill-row">
          <span>{r["display"]}<span class="bill-row-sub">{slot}</span></span>
          <span class="bill-row-amt">{"—" if r["price"]==0 else fmt(r["price"])}</span>
        </div>""" for slot, r in cnn_results.items())

        egg_row = f"""
        <div class="bill-egg-row">
          <span>Trứng thêm ×{extra_eggs}</span>
          <span class="bill-row-amt" style="color:#C04B10">+{fmt(egg_charge)}</span>
        </div>""" if egg_charge > 0 else ""

        st.markdown(f"""
        <div class="bill">
          <div class="bill-head">
            <div class="bill-head-icon">🧾</div>
            <div class="bill-head-title">Chi tiết thanh toán</div>
          </div>
          {rows}{egg_row}
          <div class="bill-footer">
            <div class="bill-footer-label">Tổng cộng</div>
            <div class="bill-footer-total">{fmt(total)}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.success(f"✅  **{fmt(total)}** — Thanh toán thành công")

elif not tray_image:
    st.markdown("""
    <div style="margin-top:8px;padding:20px 24px;background:#FFFDF7;border:1px dashed #E2DFD5;border-radius:14px;text-align:center">
      <div style="font-size:28px;margin-bottom:8px">📷</div>
      <div style="font-size:14px;font-weight:500;color:#1C1B18;margin-bottom:4px">Tải ảnh khay cơm lên để bắt đầu</div>
      <div style="font-size:12px;color:#8A887E">Hỗ trợ JPG, JPEG, PNG · hoặc chụp trực tiếp bằng webcam</div>
    </div>""", unsafe_allow_html=True)