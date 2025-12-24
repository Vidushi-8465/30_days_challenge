import cv2
import numpy as np

def generate_color_wheel(colors, size=400):
    """
    Draws a circular color wheel using dominant colors
    """
    wheel = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 2 - 10
    angle_step = 360 // len(colors)

    for i, color in enumerate(colors):
        start_angle = i * angle_step
        end_angle = start_angle + angle_step

        # ✅ FIX: Convert NumPy array to tuple of ints (BGR)
        bgr_color = tuple(int(c) for c in color[::-1])

        cv2.ellipse(
            wheel,
            (center, center),
            (radius, radius),
            0,
            start_angle,
            end_angle,
            bgr_color,
            -1
        )

    return wheel
