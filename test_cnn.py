import cv2
import numpy as np
from collections import deque, Counter
import mediapipe as mp
from keras.models import load_model

IMG_SIZE = 128
SMOOTHING_FRAMES = 8
CONF_THRESHOLD = 0.6

model = load_model("gesture_cnn.h5")

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def crop_hand(frame, landmarks, pad=30):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

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
        landmarks = result.multi_hand_landmarks[0].landmark
        hand_img = crop_hand(frame, landmarks)

        if hand_img.size != 0:
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            gray = np.expand_dims(gray, axis=0)          

            preds = model.predict(gray, verbose=0)
            idx = np.argmax(preds)
            raw_label = class_names[idx]
            raw_conf = float(preds[0][idx])

            if raw_conf > CONF_THRESHOLD:
                pred_buffer.append(raw_label)
                label = Counter(pred_buffer).most_common(1)[0][0]
                confidence = raw_conf
            else:
                pred_buffer.clear()

        for lm in landmarks:
            cx = int(lm.x * frame.shape[1])
            cy = int(lm.y * frame.shape[0])
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

    cv2.imshow("CNN HAND DETECTION", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
