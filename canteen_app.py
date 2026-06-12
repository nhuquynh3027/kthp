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
        st.warning(f"Không tải được CNN ({e})")
        return None

@st.cache_resource(show_spinner="Đang tải YOLO…")
def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        return _YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.warning(f"Không tải được YOLO ({e})")
        return None

def crop_compartment(img_np, region):
    h, w = img_np.shape[:2]
    x  = int(region[0] * w);  y  = int(region[1] * h)
    cw = int(region[2] * w);  ch = int(region[3] * h)
    return img_np[y:y+ch, x:x+cw]

def predict_dish(model, crop_np, class_names):
    if model is None:
        idx = np.random.randint(0, len(class_names))
        return class_names[idx], float(np.random.uniform(0.60, 0.97))
    resized = cv2.resize(crop_np, (IMG_SIZE, IMG_SIZE))
    inp     = np.expand_dims(resized.astype("float32") / 255.0, 0)
    preds   = model.predict(inp, verbose=0)[0]
    idx     = int(np.argmax(preds))
    name    = class_names[idx] if idx < len(class_names) else "không rõ"
    return name, float(preds[idx])

def count_eggs_yolo(yolo_model, img_np):
    annotated = img_np.copy()
    if yolo_model is None:
        count = np.random.randint(1, 4)
        h, w  = img_np.shape[:2]
        for i in range(count):
            x1 = np.random.randint(0, w//2);  y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(60, 120); y2 = y1 + np.random.randint(60, 120)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255, 200, 0), 3)
            cv2.putText(annotated, f"egg {i+1}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)
        return count, annotated
    results = yolo_model(img_np, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    count   = 0
    for box in results.boxes:
        cls_id = int(box.cls[0]); count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        label = YOLO_CLASS_NAMES.get(cls_id, f"egg cls{cls_id}")
        color = (255, 200, 0) if cls_id == 1 else (200, 150, 0)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
        cv2.putText(annotated, f"{label} {conf:.0%}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return count, annotated

def fmt_vnd(amount: int) -> str:
    return f"{amount:,}₫".replace(",", ".")

# ─────────────────────────────────────────
#  PAGE CONFIG & CSS
# ─────────────────────────────────────────
st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: #F7F6F3;
  color: #1A1A18;
}

/* ── Topbar ── */
.topbar {
  display: flex; align-items: center; gap: 16px;
  padding: 14px 20px;
  background: #1A1A18;
  border-radius: 14px;
  margin-bottom: 24px;
}
.topbar-logo {
  width: 36px; height: 36px; border-radius: 8px;
  background: #2E2E2B;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; flex-shrink: 0;
}
.topbar-name { font-size: 15px; font-weight: 500; color: #F7F6F3; line-height: 1.2; }
.topbar-desc { font-size: 12px; color: #6B6B68; margin-top: 1px; }
.topbar-badge {
  margin-left: auto;
  background: #2E2E2B; color: #9B9B98;
  font-size: 11px; font-weight: 500;
  letter-spacing: 0.04em; padding: 4px 10px;
  border-radius: 100px; border: 1px solid #3A3A37;
  flex-shrink: 0;
}

/* ── Upload zone ── */
.upload-zone {
  background: #fff;
  border: 1.5px dashed #D9D8D4;
  border-radius: 12px;
  padding: 28px 20px;
  text-align: center;
  margin-bottom: 12px;
}
.upload-zone-icon { font-size: 28px; margin-bottom: 8px; }
.upload-zone-title { font-size: 14px; font-weight: 500; color: #1A1A18; margin-bottom: 4px; }
.upload-zone-sub { font-size: 12px; color: #9B9B98; }

/* ── Section eyebrow ── */
.eyebrow {
  font-size: 11px; font-weight: 500;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #9B9B98; margin-bottom: 12px;
}

/* ── Dish card ── */
.dish-wrap {
  background: #fff;
  border: 1px solid #E8E7E3;
  border-radius: 12px;
  overflow: hidden;
}
.dish-img-wrap { aspect-ratio: 4/3; overflow: hidden; }
.dish-img-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
.dish-body { padding: 10px 12px 12px; }
.dish-slot-tag {
  display: inline-block;
  font-size: 10px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: #9B9B98;
  background: #F7F6F3; border-radius: 4px;
  padding: 2px 6px; margin-bottom: 6px;
}
.dish-title { font-size: 13px; font-weight: 500; color: #1A1A18; line-height: 1.3; margin-bottom: 8px; }
.conf-bar { background: #F0EFE9; height: 2px; border-radius: 1px; margin-bottom: 3px; }
.conf-fill { height: 2px; border-radius: 1px; }
.conf-pct { font-size: 11px; color: #B0AFA9; margin-bottom: 6px; }
.dish-price-tag {
  font-family: 'DM Mono', monospace;
  font-size: 13px; font-weight: 500; color: #2D7A3A;
}

/* ── Egg stat ── */
.egg-stat {
  background: #FFF9EE;
  border: 1px solid #F0D98A;
  border-radius: 12px;
  padding: 16px 18px;
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 12px;
}
.egg-num {
  font-family: 'DM Mono', monospace;
  font-size: 36px; font-weight: 500;
  color: #7A4F00; line-height: 1; flex-shrink: 0;
}
.egg-sub { font-size: 12px; color: #7A4F00; font-weight: 500; margin-bottom: 2px; }
.egg-note { font-size: 11px; color: #A07430; }
.egg-extra-pill {
  margin-left: auto; flex-shrink: 0;
  background: #FDE68A; color: #7A4F00;
  font-family: 'DM Mono', monospace;
  font-size: 12px; font-weight: 500;
  padding: 5px 10px; border-radius: 8px;
}

/* ── Bill ── */
.bill-wrap {
  background: #fff;
  border: 1px solid #E8E7E3;
  border-radius: 12px;
  overflow: hidden;
}
.bill-head {
  padding: 12px 16px;
  border-bottom: 1px solid #F0EFE9;
  font-size: 11px; font-weight: 500;
  letter-spacing: 0.1em; text-transform: uppercase; color: #9B9B98;
}
.bill-line {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 16px;
  border-bottom: 1px solid #F7F6F3;
  font-size: 13px; color: #3D3D3A;
}
.bill-line-slot { font-size: 11px; color: #B0AFA9; margin-left: 4px; }
.bill-line-amt {
  font-family: 'DM Mono', monospace;
  font-size: 12px; color: #3D3D3A; flex-shrink: 0;
}
.bill-egg-line {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 16px;
  border-bottom: 1px solid #F0D98A;
  background: #FFF9EE;
  font-size: 13px; color: #7A4F00; font-weight: 500;
}
.bill-total {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 16px;
  background: #1A1A18;
}
.bill-total-label {
  font-size: 11px; font-weight: 500;
  letter-spacing: 0.1em; text-transform: uppercase; color: #6B6B68;
}
.bill-total-amt {
  font-family: 'DM Mono', monospace;
  font-size: 18px; font-weight: 500; color: #5DE07A;
}

/* ── Streamlit tweaks ── */
.stButton > button {
  width: 100% !important;
  background: #1A1A18 !important; color: #F7F6F3 !important;
  border: none !important; border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important; font-weight: 500 !important;
  padding: 10px 20px !important;
  transition: background .15s !important;
}
.stButton > button:hover:not(:disabled) { background: #2E2E2B !important; }
.stButton > button:disabled { background: #E8E7E3 !important; color: #B0AFA9 !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: #E8E7E3 !important; border-radius: 12px !important; background: #fff !important;
}
div[data-testid="stImage"] img { border-radius: 8px; }
.stAlert { border-radius: 10px !important; font-size: 13px !important; }
hr { border: none !important; border-top: 1px solid #E8E7E3 !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()
demo = (cnn_model is None) or (yolo_model is None)

badge_html = '<span class="topbar-badge">Demo mode</span>' if demo else ""
st.markdown(f"""
<div class="topbar">
  <div class="topbar-logo">🍱</div>
  <div>
    <div class="topbar-name">Canteen Auto-Billing</div>
    <div class="topbar-desc">CNN nhận diện món · YOLO đếm trứng · Tính tiền tự động</div>
  </div>
  {badge_html}
</div>
""", unsafe_allow_html=True)

# ─── Input panel ───
col_input, col_pad = st.columns([1, 2])
with col_input:
    mode = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Chụp webcam"], horizontal=True, label_visibility="collapsed")
    tray_image = None
    if mode == "Tải ảnh lên":
        up = st.file_uploader("Chọn ảnh khay cơm", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up:
            tray_image = Image.open(up).convert("RGB")
    else:
        cam = st.camera_input("Hướng camera vào khay", label_visibility="collapsed")
        if cam:
            tray_image = Image.open(cam).convert("RGB")

    go = st.button("Nhận diện món & tính tiền", disabled=(tray_image is None))

if tray_image and not go:
    with col_input:
        st.image(tray_image, use_container_width=True)

if go and tray_image:
    img_np = np.array(tray_image)

    cnn_results = {}
    with st.spinner("Đang phân tích các ô…"):
        for slot, region in COMPARTMENTS.items():
            crop = crop_compartment(img_np, region)
            dish, conf = predict_dish(cnn_model, crop, CLASS_NAMES)
            dish_key = normalize_text(dish)
            price = PRICE_MAP.get(dish_key, 0)
            display = DISPLAY_NAMES.get(dish_key, dish)
            cnn_results[slot] = dict(crop=crop, dish=dish, display=display, conf=conf, price=price)

    has_thit_kho_trung = any(normalize_text(r["dish"]) == "thịt kho trứng" for r in cnn_results.values())

    egg_count, annotated_np = 0, img_np.copy()
    if has_thit_kho_trung:
        with st.spinner("Đang đếm trứng…"):
            egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    base_eggs  = 1 if has_thit_kho_trung else 0
    extra_eggs = max(0, egg_count - base_eggs)
    egg_charge = extra_eggs * EGG_SURCHARGE

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="eyebrow">Các ô trong khay</div>', unsafe_allow_html=True)

        def render_dish_card(col, slot, r):
            pct = int(r["conf"] * 100)
            bar_col = "#2D7A3A" if pct >= 80 else "#C07A10" if pct >= 60 else "#C03030"
            price_str = fmt_vnd(r["price"]) if r["price"] else "—"
            with col:
                crop_img = Image.fromarray(r["crop"])
                st.image(crop_img, use_container_width=True)
                st.markdown(f"""
                <div style="padding:2px 0 10px;">
                  <span class="dish-slot-tag">{slot}</span>
                  <div class="dish-title">{r["display"]}</div>
                  <div class="conf-bar"><div class="conf-fill" style="width:{pct}%;background:{bar_col}"></div></div>
                  <div class="conf-pct">{pct}% tin cậy</div>
                  <div class="dish-price-tag">{price_str}</div>
                </div>""", unsafe_allow_html=True)

        row1 = st.columns(2, gap="small")
        for col, slot in zip(row1, ["Top-Left", "Top-Right"]):
            render_dish_card(col, slot, cnn_results[slot])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        row2 = st.columns(3, gap="small")
        for col, slot in zip(row2, ["Bottom-Left", "Bottom-Center", "Bottom-Right"]):
            render_dish_card(col, slot, cnn_results[slot])

        if has_thit_kho_trung:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Phát hiện trứng — YOLO</div>', unsafe_allow_html=True)
            st.image(annotated_np, caption=f"Phát hiện {egg_count} quả trứng", use_container_width=True)

    with right_col:
        if has_thit_kho_trung:
            surcharge_note = f"+{extra_eggs} quả × {fmt_vnd(EGG_SURCHARGE)}" if extra_eggs > 0 else "1 trứng đã gồm trong giá"
            extra_pill = f'<span class="egg-extra-pill">+{fmt_vnd(egg_charge)}</span>' if egg_charge > 0 else ""
            st.markdown(f"""
            <div class="egg-stat">
              <div class="egg-num">{egg_count}</div>
              <div>
                <div class="egg-sub">Trứng phát hiện</div>
                <div class="egg-note">{surcharge_note}</div>
              </div>
              {extra_pill}
            </div>""", unsafe_allow_html=True)

        total = sum(r["price"] for r in cnn_results.values()) + egg_charge
        rows_html = ""
        for slot, r in cnn_results.items():
            amt = fmt_vnd(r["price"]) if r["price"] > 0 else "—"
            rows_html += f"""<div class="bill-line">
              <span>{r["display"]}<span class="bill-line-slot">{slot}</span></span>
              <span class="bill-line-amt">{amt}</span></div>"""

        egg_line_html = ""
        if egg_charge > 0:
            egg_line_html = f"""<div class="bill-egg-line">
              <span>Trứng thêm ×{extra_eggs}</span>
              <span class="bill-line-amt" style="color:#7A4F00">+{fmt_vnd(egg_charge)}</span></div>"""

        st.markdown(f"""
        <div class="bill-wrap">
          <div class="bill-head">Chi tiết thanh toán</div>
          {rows_html}{egg_line_html}
          <div class="bill-total">
            <span class="bill-total-label">Tổng cộng</span>
            <span class="bill-total-amt">{fmt_vnd(total)}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.success(f"✅ Tổng tiền: **{fmt_vnd(total)}**")

elif not tray_image:
    st.info("Tải ảnh khay cơm hoặc dùng webcam để bắt đầu.")