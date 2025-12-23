import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset"     # dataset/A, dataset/B ...
MODEL_FILE = "model.pkl"
LABEL_FILE = "labels.npy"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================
# TRAINING FUNCTION
# =========================
def train_model():
    X, y = [], []
    labels = sorted(os.listdir(DATA_DIR))

    hands = mp_hands.Hands(static_image_mode=True)

    for label in labels:
        folder = os.path.join(DATA_DIR, label)
        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    X.append(landmarks)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    np.save(LABEL_FILE, labels)
    print("✅ Model trained and saved!")

# =========================
# WEBCAM PREDICTION
# =========================
def run_webcam():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    labels = np.load(LABEL_FILE)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7)

    print("🎥 Webcam started — Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                prediction = model.predict([landmarks])[0]

                cv2.putText(
                    frame,
                    prediction,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print("📦 Training model...")
        train_model()
    run_webcam()
