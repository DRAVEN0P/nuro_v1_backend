import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Data Preprocessing Function
def load_and_preprocess_images(directory, target_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, target_size)
                # Convert to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image / 255.0  # Normalize
                images.append(image.flatten())
                labels.append(label)

    return np.array(images), np.array(labels)


def predict_image(model, scaler, image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if image is not None:
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            image = image / 255.0  # Normalize
            image = image.flatten().reshape(1, -1)
            image = scaler.transform(image)
            probabilities = model.predict_proba(image)
            prediction = model.predict(image)
            # image = Image.open(image_path)
            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()
            print(f'Predicted Class: {prediction[0]}')
            print(f'Confidence for class 0: {probabilities[0][0] * 100:.2f}%')
            print(f'Confidence for class 1: {probabilities[0][1] * 100:.2f}%')
            return {
                'predicted_class': prediction[0],
                'confidence_class_0': probabilities[0][0] * 100,
                'confidence_class_1': probabilities[0][1] * 100
            }
        else:
            # Here you need to code for retaking the image
            return {
                "confidence_class_0": 0,
                "confidence_class_1": 0,
                "predicted_class": "None"
            }  # exclude this line
