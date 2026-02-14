import cv2
import time
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GESTURES = ["up", "down", "left", "right", "peace", "palm", "fist", "thumbs_up"]
SEQ_LEN = 30
SAVE_PATH = "dataset_distance"

for g in GESTURES:
    os.makedirs(os.path.join(SAVE_PATH, g), exist_ok=True)

def extract_features(lm):
    wrist = lm[0]
    ref = lm[9]

    scale = np.sqrt(
        (ref.x - wrist.x) ** 2 +
        (ref.y - wrist.y) ** 2 +
        (ref.z - wrist.z) ** 2
    )

    features = []
    for p in lm:
        features.append((p.x - wrist.x) / scale)
        features.append((p.y - wrist.y) / scale)
        features.append((p.z - wrist.z) / scale)

    return np.array(features)

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

current_gesture = None
sequence = []
recording = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        if recording:
            features = extract_features(lm)
            sequence.append(features)

            if len(sequence) == SEQ_LEN:
                count = len(os.listdir(os.path.join(SAVE_PATH, current_gesture)))
                filename = os.path.join(
                    SAVE_PATH,
                    current_gesture,
                    f"sample_{count:04d}.npy"
                )
                np.save(filename, np.array(sequence))
                print(f"Saved: {filename}")
                sequence = []
                recording = False

        for p in lm:
            cv2.circle(
                frame,
                (int(p.x * frame.shape[1]), int(p.y * frame.shape[0])),
                3,
                (255, 0, 0),
                -1
            )

    cv2.putText(
        frame,
        f"Gesture: {current_gesture if current_gesture else 'None'}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r') and current_gesture:
        print("Recording...")
        recording = True
        sequence = []

    if key >= ord('1') and key <= ord(str(len(GESTURES))):
        index = key - ord('1')
        current_gesture = GESTURES[index]
        print(f"Selected gesture: {current_gesture}")

cap.release()
cv2.destroyAllWindows()
