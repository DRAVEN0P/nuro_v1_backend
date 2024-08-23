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
from img import predict_image
from Feature_Extraction import Feature

app = Flask(__name__)

# Load your model and scaler
model = joblib.load('./logistic_model.joblib')
scaler = joblib.load('./scaler.joblib')
# Ensure you have saved the scaler
modelWriting = joblib.load('./wr_model.pkl')
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_image_face(image, target_size=(64, 64)):
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


UPLOAD_FOLDER_IMG = 'uploads/img'
os.makedirs(UPLOAD_FOLDER_IMG, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predictImg():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:

        file_path = os.path.join(UPLOAD_FOLDER_IMG, file.filename)
        file.save(file_path)
        res = predict_image(model=model, scaler=scaler, image_path=file_path)
        return jsonify(res)

    except Exception as e:
        logging.error(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ----------------------------- Voice ----------------------------------
UPLOAD_FOLDER_AUD = 'uploads/audio'
os.makedirs(UPLOAD_FOLDER_AUD, exist_ok=True)


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
    gender = request.form.get('gender').upper()
    age = int(request.form.get('age')) 

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the file
        file_path = os.path.join(UPLOAD_FOLDER_AUD, file.filename)
        file.save(file_path)

        # Process the file
        try:
            duration, meanF0, hnr, localJitter, localShimmer, mean_energy, mean_soe, mean_zcr,f1_mean, f2_mean, f3_mean, f4_mean, f5_mean = Feature(file_path,75, 300, "Hertz",gender,age)
            
            return jsonify({
                'duration': duration,
                'gender': gender,
                'age': age,
                'hnr': hnr,
                'meanF0': meanF0,
                'f1_mean': f1_mean,
                'f2_mean': f2_mean,
                'f3_mean': f3_mean,
                'f4_mean': f4_mean,
                'f5_mean': f5_mean,
                'localJitter': localJitter,
                'localShimmer': localShimmer,
            })
        except Exception as e:
            logging.error(f"Error extracting metrics: {
                          traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logging.error(f"Error processing file: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# -------------------------Writing --------------------------------------
UPLOAD_FOLDER_WR = 'uploads/writing'
os.makedirs(UPLOAD_FOLDER_WR, exist_ok=True)


def predict_image_writing(model, scaler, image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = image / 255.0  # Normalize
        image = image.flatten().reshape(1, -1)
        image = scaler.transform(image)
        probabilities = model.predict_proba(image)
        prediction = model.predict(image)
        return {
            'predicted_class': prediction[0],
            'confidence_class_0': probabilities[0][0] * 100,
            'confidence_class_1': probabilities[0][1] * 100
        }


@app.route('/predictWriting', methods=['POST'])
def predicWriting():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER_WR, file.filename)
        file.save(file_path)
        # Preprocess the image like in your preprocessing function
        wr_res = predict_image_writing(model=modelWriting,scaler=scaler,image_path=file_path)

        return jsonify(
            wr_res
        )

    except Exception as e:
        logging.error(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
