import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Face Filter", layout="centered")
st.title("Face Filter using OpenCV (Mask Overlay)")

st.info("Allow camera access ")

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_detector = load_face_detector()

mask_file = st.file_uploader("Upload a Transparent Mask (PNG)", type=["png"])

if mask_file:
    mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
    mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_UNCHANGED)
else:
    mask_img = None

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if mask_img is not None:
        for (x, y, w, h) in faces:
            mask_resized = cv2.resize(mask_img, (w, h))
            mask_rgb = mask_resized[:, :, :3]
            mask_alpha = mask_resized[:, :, 3] / 255.0

            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    mask_alpha * mask_rgb[:, :, c]
                    + (1 - mask_alpha) * frame[y:y+h, x:x+w, c]
                )

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
