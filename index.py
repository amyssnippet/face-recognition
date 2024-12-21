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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = '0'

cap = cv2.VideoCapture(cam)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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

        if name != "Unknown":
            if name not in attendance_log:
                attendance_log[name] = {"entry_time": datetime.now(), "exit_time": None, "duration": 0}
                print(f"{name} entered at {attendance_log[name]['entry_time']}")
            else:
                attendance_log[name]["exit_time"] = datetime.now()
                duration = (attendance_log[name]["exit_time"] - attendance_log[name]["entry_time"]).total_seconds()
                attendance_log[name]["duration"] = duration

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for name, log in attendance_log.items():
    print(f"Name: {name}, Entry: {log['entry_time']}, Exit: {log.get('exit_time')}, Duration: {log['duration']} seconds")
