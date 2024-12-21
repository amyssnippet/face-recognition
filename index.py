import cv2
import os
import numpy as np
from datetime import datetime

known_faces = []
known_names = []

for file in os.listdir("known_faces"):
    img = cv2.imread(f"known_faces/{file}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    known_faces.append(img)
    known_names.append(os.path.splitext(file)[0])

attendance_log = {}
active_faces = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = 'http://192.168.42.129:8080/video'
cap = cv2.VideoCapture(cam)

def calculate_duration(entry_time, exit_time):
    return (exit_time - entry_time).total_seconds()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    detected_faces = []

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        name = "Unknown"
        min_diff = float("inf")

        for i, known_face in enumerate(known_faces):
            resized_face = cv2.resize(face, (known_face.shape[1], known_face.shape[0]))
            diff = np.sum((known_face - resized_face) ** 2)
            if diff < min_diff:
                min_diff = diff
                name = known_names[i]

        detected_faces.append(name)

        if name != "Unknown":
            if name not in active_faces:
                active_faces[name] = datetime.now()
                print(f"{name} entered at {active_faces[name]}")
            if name not in attendance_log:
                attendance_log[name] = {"entry_time": active_faces[name], "exit_time": None, "duration": 0}

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    for name in list(active_faces.keys()):
        if name not in detected_faces:
            if attendance_log[name]["exit_time"] is None:
                attendance_log[name]["exit_time"] = datetime.now()
                duration = calculate_duration(attendance_log[name]["entry_time"], attendance_log[name]["exit_time"])
                attendance_log[name]["duration"] += duration
                with open("attendance_log.txt", "a") as log_file:
                    log_file.write(f"{name} - {duration} seconds\n")
                print(f"{name} logged out with duration: {duration} seconds")
                del active_faces[name]

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for name, log in attendance_log.items():
    if log["exit_time"] is None:
        log["exit_time"] = datetime.now()
        duration = calculate_duration(log["entry_time"], log["exit_time"])
        log["duration"] += duration
        with open("attendance_log.txt", "a") as log_file:
            log_file.write(f"{name} - {duration} seconds\n")
        print(f"{name} logged out with duration: {duration} seconds")

cap.release()
cv2.destroyAllWindows()
