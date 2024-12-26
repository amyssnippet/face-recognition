import cv2
import os

def generate_dataset(user_name, num_samples=200):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]
        
    cam = "http://192.168.43.1:8080/video"
    cap = cv2.VideoCapture(cam)
    img_id = 0

    if not os.path.exists('data/'):
        os.makedirs('data/')

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/{user_name}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Cropped face', face)

        if cv2.waitKey(1) == 13 or img_id == num_samples:  # 13 is the Enter key
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {img_id} samples for user {user_name}")

if __name__ == "__main__":
    user_name = input("Enter user name: ")
    generate_dataset(user_name)
