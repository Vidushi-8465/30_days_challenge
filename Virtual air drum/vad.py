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

drum_left_img = cv2.imread("images/drum1.png", cv2.IMREAD_UNCHANGED)
drum_right_img = cv2.imread("images/drum2.png", cv2.IMREAD_UNCHANGED)
piano_left_img = cv2.imread("images/piano1.png", cv2.IMREAD_UNCHANGED)
piano_right_img = cv2.imread("images/piano2.png", cv2.IMREAD_UNCHANGED)

last_play_time = 0
cooldown = 0.25  # seconds

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Virtual Air Drum & Piano Started")
print("Press 'Q' to quit")

# Helper Function
def overlay_png(background, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w, h))

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay

    return background

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)


    # Zone sizes
    drum_height = h // 3
    piano_height = h // 3

# Drums (TOP)
    frame = overlay_png(frame, drum_left_img, 0, 0, w//2, drum_height)
    frame = overlay_png(frame, drum_right_img, w//2, 0, w//2, drum_height)

# Piano (BOTTOM)
    frame = overlay_png(frame, piano_left_img, 0, h - piano_height, w//2, piano_height)
    frame = overlay_png(frame, piano_right_img, w//2, h - piano_height, w//2, piano_height)

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
