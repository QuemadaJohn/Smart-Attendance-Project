import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

KNOWN_FACES_DIR = "known_faces"
OUTPUT_DIR = "outputs/eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

y_true = []
y_pred = []

print("Starting Evaluation on Test Split (known_faces)...")
start_time = time.time()

# Loop through images in the folder
for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    true_label = os.path.splitext(filename)[0].capitalize()

    try:
        results = DeepFace.find(img_path=img_path, db_path=KNOWN_FACES_DIR,
                                enforce_detection=False, detector_backend='retinaface',
                                silent=True, model_name='VGG-Face')

        if len(results) > 0 and not results[0].empty:
            valid_results = results[0][
                results[0]['identity'].str.contains(r'\.(jpg|png|jpeg)$', case=False, regex=True)]
            if not valid_results.empty:
                path = valid_results['identity'].iloc[0]
                pred_label = os.path.splitext(os.path.basename(path))[0].capitalize()
            else:
                pred_label = "Unknown"
        else:
            pred_label = "Unknown"

    except Exception as e:
        pred_label = "Unknown"

    y_true.append(true_label)
    y_pred.append(pred_label)
    print(f"Image: {filename} | True: {true_label} | Predicted: {pred_label}")

end_time = time.time()
total_time = end_time - start_time
fps = len(y_true) / total_time

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, zero_division=0))
print(f"Speed: {fps:.2f} images/sec")

# Save Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true + y_pred)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y_true + y_pred)))
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Face Recognition")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print(f"Confusion matrix saved to {OUTPUT_DIR}")