import cv2  # used to covert BGR --> RGB
import numpy as np  # used for array reshaping and numerical operations
from sklearn.cluster import KMeans   # to detect the dominant colours we use k means algorithm

def extract_dominant_colours(image, k=6):   #image = input, k= no of dominant colours
    """
    Extract k dominant colours using K-Means clustering
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # To conver the BRG to RGB using cv2 as matplotlib and clustering works in RBG
    pixels = image.reshape((-1, 3))   # to reshape the image i.e it flattens the image and converts the pixels to row

    kmeans = KMeans(n_clusters=k, random_state=42)  # k dominant colours and random state = 42 to keep the same output at every run
    kmeans.fit(pixels) # To train k means on all image pixels

    colours = kmeans.cluster_centers_.astype(int) # To get the cluster's center which represents the dominant colours and is converted to images from 0 - 255
    labels, counts = np.unique(kmeans.labels_, return_counts=True) #Counts how many pixels belong to each colour cluster.

    sorted_idx = np.argsort(-counts)# Sorts colours by frequency i.e (most common first).
    colours = colours[sorted_idx] # To get the index of sorted colours

    return colours
