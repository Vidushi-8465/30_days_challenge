import cv2

def load_face_detector():
    """
    Loads Haar Cascade face detector
    """
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_faces(detector, gray_frame):
    """
    Detects faces in a grayscale frame
    """
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )
    return faces
