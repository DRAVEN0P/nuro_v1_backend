import os
from flask import Flask, request, jsonify
import parselmouth
from parselmouth.praat import call
from flask_cors import CORS
import logging
import traceback

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = r'C:\Users\diyam\Documents\AudioP\ServerFlask\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)

def measurePitch(voiceID, f0min, f0max, unit):
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
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, hnr, localJitter, localShimmer

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                duration, meanF0, hnr, localJitter, localShimmer = measurePitch(file_path, 75, 500, 'Hertz')
                return jsonify({
                    'duration': duration,
                    'meanF0': meanF0,
                    'hnr': hnr,
                    'localJitter': localJitter,
                    'localShimmer': localShimmer
                })
            except Exception as e:
                app.logger.error(f"Error extracting metrics: {traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500
    except Exception as e:
        app.logger.error(f"Error processing file: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
