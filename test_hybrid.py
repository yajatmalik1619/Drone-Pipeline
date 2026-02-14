import cv2
import numpy as np
import time
import math
from collections import deque, Counter
import mediapipe as mp
from keras.models import load_model

IMG_SIZE = 128
SMOOTHING_FRAMES = 8

STATIC_GESTURES = ["fist", "palm", "peace", "thumbs_up", "up", "down", "right", "left"]

model = load_model("C:\\Users\\yajat\\Code\\drone_pipeline\\gesture_cnn.h5")

with open("C:\\Users\\yajat\\Code\\drone_pipeline\\classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def index_pointing(lm, threshold=0.25):
    wrist = lm[0]
    tip = lm[8]
    mcp = lm[5]

    finger_len = math.dist(
        (tip.x, tip.y),
        (mcp.x, mcp.y)
    )

    palm_size = math.dist(
        (lm[5].x, lm[5].y),
        (lm[17].x, lm[17].y)
    )

    return finger_len / palm_size > threshold


def other_fingers_closed(lm):
    return (
        dist(lm[12], lm[0]) < dist(lm[9], lm[0]) and
        dist(lm[16], lm[0]) < dist(lm[13], lm[0]) and
        dist(lm[20], lm[0]) < dist(lm[17], lm[0])
    )

def detect_direction(lm):
    wrist = lm[0]
    tip = lm[8]

    dx = tip.x - wrist.x
    dy = tip.y - wrist.y

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "up" if dy < 0 else "down"


def crop_hand(frame, lm, pad=30):
    h, w, _ = frame.shape
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = min(max(xs) + pad, w)
    y2 = min(max(ys) + pad, h)
    return frame[y1:y2, x1:x2]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "none"
    confidence = 0.0

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        if index_pointing(lm) and other_fingers_closed(lm):
            label = detect_direction(lm)
            confidence = 1.0
            pred_buffer.clear()
        else:
            hand_img = crop_hand(frame, lm)

            if hand_img.size != 0:
                img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = np.expand_dims(img, axis=0)

                preds = model.predict(img, verbose=0)
                idx = np.argmax(preds)
                raw_label = class_names[idx]
                raw_conf = float(preds[0][idx])

                if raw_label in STATIC_GESTURES and raw_conf > 0.6:
                    pred_buffer.append(raw_label)
                    label = Counter(pred_buffer).most_common(1)[0][0]
                    confidence = raw_conf
                else:
                    label = "none"
                    confidence = raw_conf


        for p in lm:
            cx = int(p.x * frame.shape[1])
            cy = int(p.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hybrid Gesture System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
