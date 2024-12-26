from typing import Dict
import cv2
import numpy as np
from datetime import datetime

CascadesDict = Dict[str, cv2.CascadeClassifier]

def log_recognition(user_name: str, in_time: datetime, out_time: datetime):
    duration = (out_time - in_time).total_seconds()
    log_entry = f"User: {user_name}, In-Time: {in_time}, Out-Time: {out_time}, Duration: {duration:.2f} seconds\n"
    log_file_path = "recognition_log.txt"

    with open(log_file_path, "a") as log_file:
        log_file.write(log_entry)

def recognize(img: np.ndarray, clf: cv2.face.LBPHFaceRecognizer, cascades: CascadesDict, user_map: Dict[int, str], in_times: Dict[int, datetime]) -> np.ndarray:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = cascades['face'].detectMultiScale(gray_image, 1.1, 10)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_region = gray_image[y:y+h, x:x+w]

        # Predict user identity
        id, _ = clf.predict(face_region)
        user_name = user_map.get(id, "Unknown")

        if user_name != "Unknown":
            if id not in in_times:
                in_times[id] = datetime.now()  # Record in-time
            else:
                out_time = datetime.now()
                log_recognition(user_name, in_times[id], out_time)  # Log entry
                del in_times[id]  # Clear in-time once logged

            cv2.putText(img, user_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Eye detection within the face
        eyes = cascades['eyes'].detectMultiScale(face_region, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)

        # Smile detection within the face
        smiles = cascades['smile'].detectMultiScale(face_region, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(img, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 255), 2)

    return img

if __name__ == "__main__":
    # Load Haarcascade files
    cascades: CascadesDict = {
        'face': cv2.CascadeClassifier('haarcascade_frontalface_default.xml'),
        'eyes': cv2.CascadeClassifier('haarcascade_eye.xml'),
        'smile': cv2.CascadeClassifier('haarcascade_smile.xml'),
        'upperbody': cv2.CascadeClassifier('haarcascade_upperbody.xml'),
        'fullbody': cv2.CascadeClassifier('haarcascade_fullbody.xml')
    }

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read('classifier.xml')

    # Mapping IDs to user names
    user_map = {
        hash("John") % 10000: "John",
        hash("Alice") % 10000: "Alice"
    }
    in_times = {}

    cam = "http://192.168.43.1:8080/video"
    cap = cv2.VideoCapture(cam)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video. Check the camera connection.")
            break

        frame = recognize(frame, clf, cascades, user_map, in_times)
        cv2.imshow('Advanced Recognition System', frame)

        if cv2.waitKey(1) == 13:  # 13 is the Enter key
            break

    cap.release()
    cv2.destroyAllWindows()
