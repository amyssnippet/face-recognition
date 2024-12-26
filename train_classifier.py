import cv2
import numpy as np
import os

def train_classifier(data_directory):
    paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(('.jpg', '.jpeg'))]
    faces = []
    ids = []

    for image_path in paths:
        try:
            filename = os.path.basename(image_path)  # Extract filename
            # Ensure the filename follows the format: <user_name>.<image_id>.jpg
            parts = filename.split('.')
            if len(parts) < 2:
                print(f"Skipping improperly formatted file: {filename}")
                continue

            user_name = parts[0]  # Extract user name
            user_id = hash(user_name) % 10000  # Generate a unique ID for the user
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            if img is None:
                print(f"Warning: Unable to read {image_path}. Skipping.")
                continue

            faces.append(np.array(img, dtype='uint8'))
            ids.append(user_id)

        except Exception as e:
            print(f"Error processing file {image_path}: {e}")

    if len(faces) == 0 or len(ids) == 0:
        print("No valid data to train the classifier.")
        return

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()  # Initialize the recognizer
    clf.train(faces, ids)  # Train the recognizer
    clf.save('classifier.xml')  # Save the trained model
    print("Training completed and saved as classifier.xml.")

if __name__ == "__main__":
    data_directory = 'data'
    train_classifier(data_directory)
