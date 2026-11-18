import cv2
import pandas as pd
import os
from datetime import datetime
from deepface import DeepFace

# ---------- Configuration ----------
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"

os.makedirs(ATTENDANCE_DIR, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
CSV_PATH = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

# Create CSV
expected_columns = ["Name", "Date", "Time"]
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=expected_columns).to_csv(CSV_PATH, index=False)

already_marked = set()
existing = pd.read_csv(CSV_PATH)
if not existing.empty:
    already_marked.update(existing["Name"].tolist())


def mark_attendance(name):
    if name in already_marked:
        return
    now = datetime.now()
    row = {"Name": name, "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}
    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
    already_marked.add(name)
    print(f"[ATTENDANCE] {name} marked at {row['Time']}")


def main():
    print("Starting camera. Please look at the camera...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Press 'q' to quit.")

    frame_count = 0
    current_name = "Unknown"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            try:
                results = DeepFace.find(
                    img_path=frame,
                    db_path=KNOWN_FACES_DIR,
                    enforce_detection=False,
                    detector_backend = 'retinaface',  # Or 'retinaface'
                    silent=True,
                    model_name='VGG-Face'
                )

                if len(results) > 0 and not results[0].empty:
                    # FILTER: Only look at files ending in .jpg, .png, or .jpeg
                    valid_results = results[0][
                        results[0]['identity'].str.contains(r'\.(jpg|png|jpeg)$', case=False, regex=True)]

                    if not valid_results.empty:
                        path = valid_results['identity'].iloc[0]
                        filename = os.path.basename(path)
                        current_name = os.path.splitext(filename)[0].capitalize()

                        print(f"DEBUG: Found match -> {current_name}")
                        mark_attendance(current_name)
                    else:
                        current_name = "Unknown"
                else:
                    current_name = "Unknown"
            except Exception as e:
                current_name = "Unknown"

        # Draw a box and name on the screen
        color = (0, 255, 0) if current_name != "Unknown" else (0, 0, 255)
        cv2.putText(frame, f"Student: {current_name}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Marked Today: {len(already_marked)}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Student Attendance Checker", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAttendance saved to {CSV_PATH}")


if __name__ == "__main__":
    main()