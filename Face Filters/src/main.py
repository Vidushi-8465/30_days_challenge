import cv2
from face_detector import load_face_detector, detect_faces
from mask_overlay import overlay_mask

#MASK_PATH = "../Images/glasses2.png"
MASK_PATH = "../Images/dog2.png"
def main():
    cap = cv2.VideoCapture(0)
    face_detector = load_face_detector()
    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)

    if mask_img is None:
        print("Mask image not found!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_detector, gray)

        for face in faces:
            frame = overlay_mask(frame, mask_img, face)

        cv2.imshow("Face Filter - Mask Overlay", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
