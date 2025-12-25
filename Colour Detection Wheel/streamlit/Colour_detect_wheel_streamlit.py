import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="Color Detection Wheel", layout="centered")
st.title("🎨 Color Detection Wheel using Dominant Colors")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
k = st.slider("Number of Dominant Colors", 2, 10, 6)

def extract_dominant_colors(image, k):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    colors = colors[np.argsort(-counts)]
    return colors

def generate_color_wheel(colors, size=400):
    wheel = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 2 - 10
    angle_step = 360 // len(colors)

    for i, color in enumerate(colors):
        start = i * angle_step
        end = start + angle_step
        bgr = tuple(int(c) for c in color[::-1])

        cv2.ellipse(
            wheel,
            (center, center),
            (radius, radius),
            0,
            start,
            end,
            bgr,
            -1
        )
    return wheel

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", channels="BGR")

    colors = extract_dominant_colors(image, k)
    wheel = generate_color_wheel(colors)

    st.image(wheel, caption="Color Detection Wheel", channels="BGR")
