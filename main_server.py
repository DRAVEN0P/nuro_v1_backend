from flask import Flask, request, jsonify
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import joblib
import os
import logging
import traceback
import parselmouth
from parselmouth.praat import call

app = Flask(__name__)

# Load your model and scaler
model = joblib.load('./logistic_model.joblib')
scaler = joblib.load('./scaler.joblib')
# Ensure you have saved the scaler
modelWriting = joblib.load('./writingModel.joblib')


def preprocess_image(image, target_size=(64, 64)):
    # Resize the image
    image = cv2.resize(image, target_size)
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize the image
    image = image / 255.0
    # Flatten the image
    image = image.flatten().reshape(1, -1)
    # Scale the image
    image = scaler.transform(image)
    return image


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image file as bytes
        image_bytes = file.read()

        # Convert bytes to a NumPy array
        image_array = np.frombuffer(image_bytes, np.uint8)

        # Decode the image array into an image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to process the image"}), 400

        # Preprocess the image like in your preprocessing function
        processed_image = preprocess_image(image)

        # Perform prediction
        probabilities = model.predict_proba(processed_image)
        prediction = model.predict(processed_image)

        return jsonify({
            'predicted_class': prediction[0],
            'confidence_class_0': probabilities[0][0] * 100,
            'confidence_class_1': probabilities[0][1] * 100
        })

    except Exception as e:
        logging.error(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def measure_pitch(voiceID, f0min, f0max, unit):
    # Load the sound using parselmouth
    sound = parselmouth.Sound(voiceID)

    # Duration of the sound
    duration = call(sound, "Get total duration")

    # Calculate F0 (mean)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)

    # Harmonicity (HNR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Point process for jitter and shimmer
    point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    local_jitter = call(point_process, "Get jitter (local)",
                        0, 0, 0.0001, 0.02, 1.3)
    local_shimmer = call([sound, point_process],
                         "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return duration, meanF0, hnr, local_jitter, local_shimmer


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the file
        try:
            duration, meanF0, hnr, local_jitter, local_shimmer = measure_pitch(
                file_path, 75, 500, 'Hertz')
            return jsonify({
                'duration': duration,
                'meanF0': meanF0,
                'hnr': hnr,
                'localJitter': local_jitter,
                'localShimmer': local_shimmer
            })
        except Exception as e:
            logging.error(f"Error extracting metrics: {
                          traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logging.error(f"Error processing file: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/predictWriting', methods=['POST'])
def predicWriting():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image file as bytes
        image_bytes = file.read()

        # Convert bytes to a NumPy array
        image_array = np.frombuffer(image_bytes, np.uint8)

        # Decode the image array into an image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Failed to process the image"}), 400

        # Preprocess the image like in your preprocessing function
        processed_image = preprocess_image(image)

        # Perform prediction
        probabilities = modelWriting.predict_proba(processed_image)
        prediction = modelWriting.predict(processed_image)

        return jsonify({
            'predicted_class': prediction[0],
            'confidence_class_0': probabilities[0][0] * 100,
            'confidence_class_1': probabilities[0][1] * 100
        })

    except Exception as e:
        logging.error(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
