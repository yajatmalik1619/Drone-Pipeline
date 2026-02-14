import cv2
import time
import math
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

gesture_buffer = deque(maxlen=10)
position_history = deque(maxlen=15)
last_confirmed = None
motion_cooldown_time = 0
motion_cooldown = 1.0

def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def finger_open(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def thumb_open(lm):
    return abs(lm[4].x - lm[3].x) > 0.03

def fingers_state(lm):
    return {
        "thumb": thumb_open(lm),
        "index": finger_open(lm, 8, 6),
        "middle": finger_open(lm, 12, 10),
        "ring": finger_open(lm, 16, 14),
        "pinky": finger_open(lm, 20, 18),
    }

def is_fist(lm):
    closed = (
        lm[4].y > lm[0].y and
        lm[8].y > lm[6].y and
        lm[12].y > lm[10].y and
        lm[16].y > lm[14].y and
        lm[20].y > lm[18].y
    )
    return closed

def is_thumbs_up(lm, f):
    thumb_upward = lm[4].y < lm[3].y and lm[4].y < lm[2].y
    fingers_closed = not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]
    return thumb_upward and fingers_closed


def detect_static(lm, f):
    if is_thumbs_up(lm, f):
        return "thumbs_up"

    if is_fist(lm):
        return "fist"

    if f["index"] and f["middle"] and not (f["ring"] or f["pinky"]):
        return "peace"

    if all(f.values()):
        return "palm_static"

    if f["index"] and not (f["middle"] or f["ring"] or f["pinky"]):
        mcp, tip = lm[5], lm[8]
        dx, dy = tip.x - mcp.x, tip.y - mcp.y
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "up" if dy < 0 else "down"

    return None

def confirm(current):
    gesture_buffer.append(current)
    if len(gesture_buffer) < 6:
        return None
    g, c = Counter(gesture_buffer).most_common(1)[0]
    return g if g is not None and c / len(gesture_buffer) >= 0.75 else None

def detect_motion():
    global motion_cooldown_time
    if len(position_history) < 15:
        return None
    current_time = time.time()
    if current_time - motion_cooldown_time < motion_cooldown:
        return None
    x_start, y_start = position_history[0]
    x_end, y_end = position_history[-1]
    dx = x_end - x_start
    dy = y_end - y_start
    threshold = 0.07
    if abs(dx) > abs(dy):
        if dx > threshold:
            motion_cooldown_time = current_time
            position_history.clear()
            return "swipe_right"
        if dx < -threshold:
            motion_cooldown_time = current_time
            position_history.clear()
            return "swipe_left"
    else:
        if dy > threshold:
            motion_cooldown_time = current_time
            position_history.clear()
            return "swipe_down"
        if dy < -threshold:
            motion_cooldown_time = current_time
            position_history.clear()
            return "swipe_up"
    return None

def detect_flip(lm):
    wrist = lm[0]
    index = lm[5]
    pinky = lm[17]
    v1x = index.x - wrist.x
    v1y = index.y - wrist.y
    v2x = pinky.x - wrist.x
    v2y = pinky.y - wrist.y
    cross = v1x * v2y - v1y * v2x
    return "palm" if cross > 0 else "back"

base_options = python.BaseOptions(
    model_asset_path="C:\\Users\\yajat\\Code\\drone_pipeline\\hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
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

    confirmed = None
    orientation = None

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        wrist = lm[0]
        position_history.append((wrist.x, wrist.y))

        f = fingers_state(lm)
        current_static = detect_static(lm, f)
        confirmed = confirm(current_static)

        if confirmed is None:
            motion = detect_motion()
            if motion:
                confirmed = motion

        orientation = detect_flip(lm)

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
    else:
        position_history.clear()

    if last_confirmed:
        cv2.putText(
            frame,
            f"Gesture: {last_confirmed}",
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

    if orientation:
        cv2.putText(
            frame,
            f"Facing: {orientation}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
