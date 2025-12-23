import cv2
import mediapipe as mp
import numpy as np
import os

# ---------------- CONFIG ----------------
DATA_DIR = "data/landmarks"
LABEL = "A"      # change for each sign (A, B, C...)
SAMPLES = 300    # samples per sign
# ----------------------------------------

os.makedirs(os.path.join(DATA_DIR, LABEL), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        np.save(f"{DATA_DIR}/{LABEL}/{count}.npy", np.array(landmarks))
        count += 1

        cv2.putText(frame, f"Samples: {count}/{SAMPLES}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    cv2.imshow("Dataset Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()
