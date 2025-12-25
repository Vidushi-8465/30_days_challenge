import cv2    # 
import numpy as np

def generate_colour_wheel(colors, size=400):
    """
    Draws a circular color wheel using dominant colours
    """
    wheel = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 2 - 10
    angle_step = 360 // len(colors)

    for i, colour in enumerate(colors):
        start_angle = i * angle_step
        end_angle = start_angle + angle_step

        # 
        # FIX: Convert NumPy array to tuple of ints (BGR)
        bgr_colour = tuple(int(c) for c in colour[::-1])

        cv2.ellipse(
            wheel,
            (center, center),
            (radius, radius),
            0,
            start_angle,
            end_angle,
            bgr_colour,
            -1
        )

    return wheel
