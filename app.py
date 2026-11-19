import cv2
import pandas as pd
import os
import threading
import time
from datetime import datetime
from deepface import DeepFace
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

# ---------- Configuration ----------
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

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

# ---------- Global Variables ----------
camera_running = False
is_registering = False
cap = None
current_known_name = "Unknown"
current_face_box = None
last_detection_time = 0

# ---------- AI Settings ----------
MODEL_NAME = "Facenet512"  # Upgraded from VGG-Face for better accuracy
DETECTOR_BACKEND = "retinaface"  # Slower but highly accurate
DISTANCE_THRESHOLD = 0.30  # Stricter threshold for Facenet


def mark_attendance(name):
    if name in already_marked or name == "Unknown":
        return False
    now = datetime.now()
    row = {"Name": name, "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S")}
    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=False, index=False)
    already_marked.add(name)

    app.after(0, lambda: update_ui_status(f"Marked: {name} at {row['Time']}", "green"))
    app.after(0, lambda: update_marked_count())
    return True


def update_ui_status(text, color="green"):
    status_label.config(text=text, fg=color)


def update_marked_count():
    count_label.config(text=f"Total Marked Today: {len(already_marked)}")


# ---------- Preprocessing ----------
def preprocess_image(frame):
    """Standardizes the frame before sending to the AI."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 480))  # Standard size
    return resized


# ---------- Registration Logic ----------
def register_student():
    global is_registering
    name = simpledialog.askstring("Register Student", "Enter student's name:")
    if not name:
        return

    name = name.strip().capitalize()
    if not name.isalpha():
        messagebox.showerror("Error", "Please use letters only (no spaces or numbers).")
        return

    is_registering = True
    time.sleep(1)

    status_label.config(text="Get ready! Capturing in 3 seconds...", fg="orange")
    app.update()

    temp_cap = cv2.VideoCapture(0)
    if not temp_cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        status_label.config(text="Ready", fg="white")
        is_registering = False
        return

    time.sleep(1.5)

    # Countdown
    for i in range(3, 0, -1):
        status_label.config(text=f"Capturing in {i}...", fg="orange")
        ret, frame = temp_cap.read()
        if ret:
            display_frame = cv2.resize(frame, (640, 480))
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            app.video_label.imgtk = imgtk
            app.video_label.configure(image=imgtk)
            app.update()
        time.sleep(1)

    ret, frame = temp_cap.read()
    temp_cap.release()

    if ret:
        filepath = os.path.join(KNOWN_FACES_DIR, f"{name.lower()}.jpg")
        cv2.imwrite(filepath, frame)
        status_label.config(text=f"Successfully registered {name}!", fg="green")
        messagebox.showinfo("Success", f"{name} has been registered!")
    else:
        status_label.config(text="Failed to capture image.", fg="red")
        messagebox.showerror("Error", "Could not capture image.")

    is_registering = False
    app.video_label.configure(image='')
    status_label.config(text="Ready", fg="white")


# ---------- Camera Logic ----------
def run_camera():
    global cap, camera_running, is_registering
    global current_known_name, current_face_box, last_detection_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        app.after(0, lambda: update_ui_status("Error: Camera not found.", "red"))
        return

    frame_count = 0

    while camera_running and cap.isOpened():
        if is_registering:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame to save CPU
        if frame_count % 10 == 0:
            processed_frame = preprocess_image(frame)

            try:
                # DeepFace requires numpy array in BGR format
                results = DeepFace.find(
                    img_path=processed_frame,
                    db_path=KNOWN_FACES_DIR,
                    enforce_detection=False,
                    detector_backend=DETECTOR_BACKEND,
                    silent=True,
                    model_name=MODEL_NAME,
                    distance_metric='cosine'
                )

                if len(results) > 0 and not results[0].empty:
                    # Filter out non-image files (like .h5 models)
                    valid_results = results[0][
                        results[0]['identity'].str.contains(r'\.(jpg|png|jpeg)$', case=False, regex=True)]

                    if not valid_results.empty:
                        best_match_distance = valid_results['distance'].iloc[0]

                        if best_match_distance < DISTANCE_THRESHOLD:
                            path = valid_results['identity'].iloc[0]
                            filename = os.path.basename(path)
                            current_known_name = os.path.splitext(filename)[0].capitalize()

                            # Get bounding box from the DeepFace result
                            source_x = valid_results['source_x'].iloc[0]
                            source_y = valid_results['source_y'].iloc[0]
                            source_w = valid_results['source_w'].iloc[0]
                            source_h = valid_results['source_h'].iloc[0]

                            current_face_box = (source_x, source_y, source_w, source_h)
                            last_detection_time = time.time()

                            mark_attendance(current_known_name)
                        else:
                            current_known_name = "Unknown"
                            current_face_box = None
                    else:
                        current_known_name = "Unknown"
                        current_face_box = None
                else:
                    current_known_name = "Unknown"
                    current_face_box = None

            except Exception as e:
                current_known_name = "Unknown"
                current_face_box = None

        # Draw the box on the frame
        display_frame = cv2.resize(frame, (640, 480))
        scale_x = display_frame.shape[1] / processed_frame.shape[1]
        scale_y = display_frame.shape[0] / processed_frame.shape[0]

        # Only draw if a face was detected recently (within 1 second)
        if current_face_box and (time.time() - last_detection_time < 1.0):
            x, y, w, h = current_face_box
            # Scale coordinates to match the display frame size
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

            color = (0, 255, 0) if current_known_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, current_known_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # If no box, still show the status text
            color = (0, 255, 0) if current_known_name != "Unknown" else (0, 0, 255)
            cv2.putText(display_frame, f"Student: {current_known_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,
                        2)

        # Convert and update Tkinter UI
        cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update UI safely (ignoring errors if user clicks X)
        try:
            app.video_label.imgtk = imgtk
            app.video_label.configure(image=imgtk)
        except:
            pass

        frame_count += 1

    if cap:
        cap.release()
    try:
        app.video_label.configure(image='')
    except:
        pass


def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
        update_ui_status("Camera Started. Scanning...", "blue")
        threading.Thread(target=run_camera, daemon=True).start()


def stop_camera():
    global camera_running
    camera_running = False
    update_ui_status("Camera Stopped.", "red")


def view_attendance():
    try:
        os.startfile(CSV_PATH)
    except Exception as e:
        messagebox.showerror("Error", f"Could not open file: {e}")


# ---------- UI Setup ----------
app = tk.Tk()
app.title("Smart Attendance System")
app.geometry("750x680")
app.configure(bg="#2c3e50")

title_label = tk.Label(app, text="Face Recognition Attendance", font=("Helvetica", 18, "bold"), bg="#2c3e50",
                       fg="white")
title_label.pack(pady=10)

video_frame = tk.Frame(app, bg="black", width=640, height=480)
video_frame.pack(pady=10)
app.video_label = tk.Label(video_frame, bg="black")
app.video_label.pack()

btn_frame = tk.Frame(app, bg="#2c3e50")
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="▶ Start Camera", font=("Helvetica", 12), bg="#2ecc71", fg="white",
                      command=start_camera)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="⏹ Stop Camera", font=("Helvetica", 12), bg="#e74c3c", fg="white",
                     command=stop_camera)
stop_btn.grid(row=0, column=1, padx=10)

csv_btn = tk.Button(btn_frame, text="📄 View CSV", font=("Helvetica", 12), bg="#3498db", fg="white",
                    command=view_attendance)
csv_btn.grid(row=0, column=2, padx=10)

register_btn = tk.Button(btn_frame, text="➕ Register New Student", font=("Helvetica", 12, "bold"), bg="#f39c12",
                         fg="white", command=register_student)
register_btn.grid(row=0, column=3, padx=10)

status_label = tk.Label(app, text="Ready", font=("Helvetica", 12), bg="#2c3e50", fg="white")
status_label.pack(pady=5)

count_label = tk.Label(app, text=f"Total Marked Today: {len(already_marked)}", font=("Helvetica", 12, "bold"),
                       bg="#2c3e50", fg="#f1c40f")
count_label.pack(pady=5)

if __name__ == "__main__":
    app.mainloop()