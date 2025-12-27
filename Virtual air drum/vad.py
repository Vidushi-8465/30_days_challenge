import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ---------------- INITIALIZE ----------------
pygame.mixer.init()
pygame.mixer.set_num_channels(8)

# Load sounds
drum1 = pygame.mixer.Sound("sounds/drum1.wav")
drum2 = pygame.mixer.Sound("sounds/drum2.wav")
piano1 = pygame.mixer.Sound("sounds/piano1.wav")
piano2 = pygame.mixer.Sound("sounds/piano2.wav")

last_play_time = 0
cooldown = 0.25  # seconds

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Virtual Air Drum & Piano Started")
print("Press 'Q' to quit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw Zones
    cv2.rectangle(frame, (0, 0), (w//2, h//3), (255, 0, 0), 2)
    cv2.rectangle(frame, (w//2, 0), (w, h//3), (0, 255, 0), 2)
    cv2.rectangle(frame, (0, h*2//3), (w//2, h), (0, 0, 255), 2)
    cv2.rectangle(frame, (w//2, h*2//3), (w, h), (255, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index_tip = hand.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

            current_time = time.time()
            if current_time - last_play_time > cooldown:
                # Drum
                if y < h // 3:
                    if x < w // 2:
                        drum1.play()
                    else:
                        drum2.play()
                    last_play_time = current_time

                # Piano
                elif y > h * 2 // 3:
                    if x < w // 2:
                        piano1.play()
                    else:
                        piano2.play()
                    last_play_time = current_time

    cv2.imshow(" Virtual Air Drum & Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
