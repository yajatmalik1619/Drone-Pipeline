import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

def detect_gesture(landmarks, fingers_data, w, h, threshold=0.15):
    index = fingers_data['INDEX']
    middle = fingers_data['MIDDLE']
    ring = fingers_data['RING']
    pinky = fingers_data['PINKY']
    thumb = fingers_data['THUMB']
    wrist = landmarks[0]
    if index and middle:
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        # dx = index_tip.x - middle_tip.x
        # dy = index_tip.y - middle_tip.y
        # distance = math.sqrt((dx)**2 + (dy**2))
        # if distance > 0.5:
        # dx = index_tip.x - wrist.x
        # dy = index_tip.y - wrist.y

        # if dy > threshold:
        return 'UP MORE FORCE'
    elif ring and middle:
        return 'DOWN MORE FORCE'
    elif index:
        index_tip = landmarks[8]
        wrist = landmarks[0]
        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y

        if abs(dx) > abs(dy):
            if dx > threshold:
                return "RIGHT"
            elif dx < -threshold:
                return "LEFT"
        else:
            if dy < -threshold:
                return "UP"
            elif dy > threshold:
                return "DOWN"
    elif middle:
        return 'FUCK OFF'
    if fist(landmarks):
        return 'FIST'
    return None

def detect_fingers(hand_landmarks):
    fingers = {
        "thumb":  (4, 2),
        "index":  (8, 5),
        "middle": (12, 9),
        "ring":   (16, 13),
        "pinky":  (20, 17)
    }

    status = {}
    for finger, (tip, mcp) in fingers.items():
        status[finger] = finger_is_open(hand_landmarks, tip, mcp)
    return status

def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 +(a.y - b.y) ** 2 )

def finger_is_open(landmarks, tip, mcp):
    wrist = landmarks[0]
    return dist(landmarks[tip], wrist) > dist(landmarks[mcp], wrist)

def thumb_is_open(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    index_mcp = landmarks[5]

    return abs(thumb_tip.x - index_mcp.x) > abs(thumb_ip.x - index_mcp.x)

def fist(landmarks, threshold = 0.25):
    wrist = landmarks[0]
    fingertips = [4,8,12,16,20]
    total_distance = 0
    for i in fingertips:
        distance = math.sqrt((landmarks[i].x-wrist.x)**2 + (landmarks[i].y - wrist.y)**2)
        total_distance += distance
    total_distance /= 5 
    if total_distance < threshold:
        return True

base_options = python.BaseOptions(
    model_asset_path="C:\\Users\\yajat\\Code\\drone_pipeline\\hand_landmarker.task"
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
    h, w, _ = frame.shape

    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=frame
)


    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]

        center = (int(np.mean(xs)), int(np.mean(ys)))
        cv2.circle(frame, center, 6, (0, 255, 0), -1)

        fingers = detect_fingers(landmarks= result.hand_landmarks[0])
        thumb = thumb_is_open(landmarks = result.hand_landmarks[0])
        gesture = detect_gesture(landmarks=result.hand_landmarks[0], fingers_data = fingers ,w=w,h=h)

        if gesture:
            print(f"Gesture Detected: {gesture}")
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        prev_center = center

        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

    cv2.imshow("Drone Gesture Demo (MediaPipe Tasks)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
