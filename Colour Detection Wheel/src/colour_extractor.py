import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_colors(image, k=6):
    """
    Extract k dominant colors using K-Means clustering
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # Sort colors by frequency
    sorted_idx = np.argsort(-counts)
    colors = colors[sorted_idx]

    return colors
