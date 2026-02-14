import cv2
import time
import math
import os
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_DIR = "dataset"

gesture_buffer = deque(maxlen=10)
last_confirmed = None

def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def finger_open(lm, tip, mcp):
    return dist(lm[tip], lm[0]) > dist(lm[mcp], lm[0])

def thumb_open(lm):
    return dist(lm[4], lm[0]) > dist(lm[3], lm[0]) * 1.2

def fingers_state(lm):
    return {
        "thumb": thumb_open(lm),
        "index": finger_open(lm, 8, 5),
        "middle": finger_open(lm, 12, 9),
        "ring": finger_open(lm, 16, 13),
        "pinky": finger_open(lm, 20, 17),
    }

def is_fist(f):
    return (
        not f["thumb"] and
        not f["index"] and
        not f["middle"] and
        not f["ring"] and
        not f["pinky"]
    )

def is_thumbs_up(lm, f):
    return (
        f["thumb"] and
        not f["index"] and
        not f["middle"] and
        not f["ring"] and
        not f["pinky"] and
        lm[4].y < lm[0].y
    )

def detect_gesture(lm, f):
    if is_thumbs_up(lm, f):
        return "thumbs_up"

    if f["index"] and not (f["middle"] or f["ring"] or f["pinky"]):
        mcp, tip = lm[5], lm[8]
        dx, dy = tip.x - mcp.x, tip.y - mcp.y
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "up" if dy < 0 else "down"

    if f["index"] and f["middle"] and not (f["ring"] or f["pinky"]):
        return "peace"

    if all(f.values()):
        return "palm"

    if is_fist(f):
        return "fist"

    return None

def confirm(current):
    if current is None:
        gesture_buffer.clear()
        return None
    gesture_buffer.append(current)
    if len(gesture_buffer) < 6:
        return None
    g, c = Counter(gesture_buffer).most_common(1)[0]
    return g if c / len(gesture_buffer) >= 0.75 else None

def crop_hand(frame, lm, pad=30):
    h, w, _ = frame.shape
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = min(max(xs) + pad, w)
    y2 = min(max(ys) + pad, h)
    return frame[y1:y2, x1:x2]

base_options = python.BaseOptions(
    model_asset_path= "hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    lm = None
    confirmed = None

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        f = fingers_state(lm)
        current = detect_gesture(lm, f)
        confirmed = confirm(current)

        if confirmed:
            last_confirmed = confirmed

        for p in lm:
            cv2.circle(
                frame,
                (int(p.x * frame.shape[1]), int(p.y * frame.shape[0])),
                3,
                (255, 0, 0),
                -1
            )

    if last_confirmed:
        cv2.putText(
            frame,
            f"READY: {last_confirmed}  (press C)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "NO GESTURE",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    cv2.imshow("Manual Gesture Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and last_confirmed and lm is not None:
        folder = os.path.join(DATASET_DIR, last_confirmed)
        if os.path.isdir(folder):
            img = crop_hand(frame, lm)
            path = os.path.join(folder, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, img)
            print("saved:", path)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
