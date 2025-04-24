import cv2
import mediapipe as mp
import pyautogui
import math
import time
from collections import deque


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

volume_history = deque(maxlen=5)
prev_volume_send_time = time.time()
prev_volume = None

# Gesture cooldowns
last_skip_time = 0
last_pause_time = 0
gesture_cooldown = 1.0  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            coords = [(int(p.x * w), int(p.y * h)) for p in lm]

            if label == 'Left':
                # VOLUME CONTROL
                thumb, index, middle = coords[4], coords[8], coords[12]
                stop = distance(thumb, middle)
                d = distance(thumb, index)
                raw_volume = int(min(max((d - 20) / 160 * 100, 0), 100))
                volume_history.append(raw_volume)
                smooth_volume = int(sum(volume_history) / len(volume_history))

                now = time.time()
                if now - prev_volume_send_time > 0.5:
                    prev_volume_send_time = now
                    print(f"Smooth Volume: {smooth_volume}%")

                    if prev_volume is None:
                        prev_volume = smooth_volume

                    diff = smooth_volume - prev_volume
                    if abs(diff) >= 5:
                        if diff > 0:
                            pyautogui.press("volumeup", presses=int(diff / 2))
                        else:
                            pyautogui.press("volumedown", presses=int(abs(diff) / 2))
                        prev_volume = smooth_volume
                
                if stop - 20 < 0:
                    print("Middle/Thumb Connection → Exiting app")
                    cap.release()
                    cv2.destroyAllWindows()

            elif label == 'Right':
                now = time.time()

                index_tip_y = coords[8][1]
                index_base_y = coords[6][1]
                middle_tip_y = coords[12][1]
                middle_base_y = coords[10][1]
                ring_tip_y = coords[16][1]
                ring_base_y = coords[14][1]
                pinky_tip_y = coords[20][1]
                pinky_base_y = coords[18][1]

                if index_tip_y < index_base_y - 25 and now - last_skip_time > gesture_cooldown and middle_tip_y > middle_base_y - 10:
                    print("Index finger raised → Skip track")
                    pyautogui.hotkey("ctrl", "right")
                    last_skip_time = now
        
                if pinky_tip_y < pinky_base_y - 25 and now - last_skip_time > gesture_cooldown and middle_tip_y > middle_base_y - 10:
                    print("Pinky raised → Previous track/Restart track")
                    pyautogui.hotkey("ctrl", "left")
                    last_skip_time = now

                # Play/Pause if open palm
                fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
                bases = [6, 10, 14, 18]
                fingers_up = all(coords[tip][1] < coords[base][1] - 20 for tip, base in zip(fingertips, bases))

                thumb_tip_x = coords[4][0]
                thumb_base_x = coords[2][0]
                thumb_open = abs(thumb_tip_x - thumb_base_x) > 40

                if fingers_up and thumb_open and now - last_pause_time > gesture_cooldown:
                    print("Open Palm → Play/Pause")
                    pyautogui.press("playpause")
                    last_pause_time = now

    cv2.imshow("Spotify Hand Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
