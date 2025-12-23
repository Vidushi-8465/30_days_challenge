import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

with open("labels.txt") as f:
    labels = f.read().splitlines()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

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

        prediction = model.predict(np.array([landmarks]), verbose=0)
        sign = labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        cv2.putText(frame, f"{sign} ({confidence:.2f})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
