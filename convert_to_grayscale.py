import cv2
import os

directory = "dataset"
directory_gs = "dataset_grayscale"

for root, dirs, files in os.walk(directory):
    rel_path = os.path.relpath(root, directory)
    save_dir = os.path.join(directory_gs, rel_path)
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(root, file)
        path_to_write = os.path.join(save_dir, file)

        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path_to_write, gray)
