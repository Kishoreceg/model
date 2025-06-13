import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Underwater Enhancer & Fish/Coral Detector",
                   layout="wide",
                   page_icon="🌊")

st.title("🌊 Underwater Image Enhancer & Detector")
st.write(
    "Upload a blurry underwater photo. The app will enhance it, then try to highlight **fish** (blue boxes) "
    "and **coral** (red boxes) using lightweight colour/shape heuristics."
)

uploaded = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])

# ---------- Core image‑processing helpers ----------
def enhance_image(bgr):
    """CLAHE on L‑channel + simple white balance + mild sharpening."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    bgr_eq = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    wb = cv2.xphoto.createSimpleWB()
    bgr_wb = wb.balanceWhite(bgr_eq)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(bgr_wb, -1, kernel)
    return sharp

def detect_coral(bgr):
    """Very naive: redness mask ⇒ contours."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,  50,  50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 800]
    return boxes

def detect_fish(bgr):
    """Very naive: brightness blobs ⇒ contours."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(c) for c in contours
        if 500 < cv2.contourArea(c) < 50000
    ]
    return boxes

def draw_boxes(bgr, fish_boxes, coral_boxes):
    out = bgr.copy()
    for x, y, w, h in fish_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)   # blue
    for x, y, w, h in coral_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)   # red
    return out

# ---------- Main workflow ----------
if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    enhanced = enhance_image(bgr)
    fish_boxes  = detect_fish(enhanced)
    coral_boxes = detect_coral(enhanced)
    detected    = draw_boxes(enhanced, fish_boxes, coral_boxes)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Enhanced")
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col3:
        st.subheader("Detected")
        st.image(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.markdown(
        f"- **Fish detected**: {len(fish_boxes)}\n"
        f"- **Coral regions**: {len(coral_boxes)}"
    )

    st.info(
        "The detection logic here is deliberately lightweight for demo purposes. "
        "For production‑quality accuracy consider training a small YOLO/SSD model "
        "on an underwater dataset such as *Sea‑Thru*, *ROV fish DB*, or *CoralNet*."
    )
