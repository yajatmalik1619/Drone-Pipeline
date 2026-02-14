import cv2
import mediapipe as mp
import numpy as np
import os

INPUT_DIR = "dataset_grayscale"
OUTPUT_DIR = "dataset_mask"

IMG_SIZE = 128

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4
)

for root, dirs, files in os.walk(INPUT_DIR):

    rel_path = os.path.relpath(root, INPUT_DIR)
    save_dir = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(save_dir, exist_ok=True)

    for file in files:

        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(root, file)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            landmarks = result.multi_hand_landmarks[0].landmark

            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            box_width = x_max - x_min
            box_height = y_max - y_min

            pad_x = int(0.15 * box_width)
            pad_y = int(0.15 * box_height)

            x1 = max(x_min - pad_x, 0)
            y1 = max(y_min - pad_y, 0)
            x2 = min(x_max + pad_x, w)
            y2 = min(y_max + pad_y, h)

            cropped = img[y1:y2, x1:x2]

        else:
            cropped = img

        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, resized)
