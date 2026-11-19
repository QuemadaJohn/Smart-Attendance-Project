import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from sklearn.metrics import classification_report

KNOWN_FACES_DIR = "../known_faces"
ATTENDANCE_DIR = "../attendance"

y_true = []
y_pred = []

print("Running Evaluation...")
# We test the known_faces folder against itself to check for confusion
for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    true_label = os.path.splitext(filename)[0].capitalize()

    try:
        results = DeepFace.find(img_path=img_path, db_path=KNOWN_FACES_DIR,
                                enforce_detection=True, detector_backend='retinaface',
                                silent=True, model_name='VGG-Face', distance_metric='cosine')

        if len(results) > 0 and not results[0].empty:
            best_match_distance = results[0]['distance'].iloc[0]
            if best_match_distance < 0.35:
                path = results[0]['identity'].iloc[0]
                pred_label = os.path.splitext(os.path.basename(path))[0].capitalize()
            else:
                pred_label = "Unknown"
        else:
            pred_label = "Unknown"
    except:
        pred_label = "Error"

    y_true.append(true_label)
    y_pred.append(pred_label)

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, zero_division=0))