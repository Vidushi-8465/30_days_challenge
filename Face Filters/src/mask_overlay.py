import cv2
import numpy as np

def overlay_mask(frame, mask_img, face):
    """
    Overlays a transparent mask on detected face
    """
    x, y, w, h = face

    mask_resized = cv2.resize(mask_img, (w, h))

    mask_rgb = mask_resized[:, :, :3]
    mask_alpha = mask_resized[:, :, 3] / 255.0

    for c in range(3):
        frame[y:y+h, x:x+w, c] = (
            mask_alpha * mask_rgb[:, :, c] +
            (1 - mask_alpha) * frame[y:y+h, x:x+w, c]
        )

    return frame
