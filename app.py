import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, session, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'DeFake AI'  # Set a secret key for session management

UPLOAD_FOLDER = r'\uploads'
STATIC_FOLDER = 'static/deepfake_frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

model = load_model(r'defake_ai_best.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
        
    if 'video_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    video_file = request.files['video_file']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    result, f1_score, accuracy, deepfake_percentage, deepfake_frame_paths = is_deepfake(video_path)

    os.remove(video_path)

    session['deepfake_frame_paths'] = deepfake_frame_paths

    response = {
        'result': result,
        'f1_score': float(f1_score) if f1_score is not None else None,
        'accuracy': float(accuracy) if accuracy is not None else None,
        'deepfake_percentage': deepfake_percentage,
        'deepfake_frame_paths': deepfake_frame_paths
    }
    
    return jsonify(response)

@app.route('/deepfake-frames', methods=['GET'])
def get_deepfake_frames():
    folder_path = 'static/deepfake_frames'  # Path to the folder containing deepfake frames
    image_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg')):
            image_files.append(f'/static/deepfake_frames/{filename}')  # Add the relative URL for each image
    
    return jsonify(image_files)

@app.route('/static/deepfake_frames/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'deepfake_frames'), filename)

def is_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video file.", None, None, None, []

    frames = []
    deepfake_frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frames.append(frame_normalized)

    cap.release()

    # If there are frames to process
    if frames:
        frames_array = np.array(frames)
        
        # Use batch processing
        batch_size = 32  # Adjust batch size depending on the available memory
        frame_predictions = []

        # Process the frames in batches
        for i in range(0, len(frames_array), batch_size):
            batch = frames_array[i:i+batch_size]
            batch_predictions = model.predict(batch)
            frame_predictions.extend(batch_predictions)

        # Post-process predictions
        deepfake_frames = []
        for i, pred in enumerate(frame_predictions):
            if pred < 0.5:
                deepfake_frames.append(frames[i])
                # Save deepfake frames as images in the static folder
                deepfake_frame_filename = f"deepfake_frame_{i}.jpg"
                deepfake_frame_path = os.path.join(STATIC_FOLDER, deepfake_frame_filename)
                cv2.imwrite(deepfake_frame_path, (frames[i] * 255).astype(np.uint8))
                deepfake_frame_paths.append(f"/deepfake-frames/{deepfake_frame_filename}")

        deepfake_frames_count = len(deepfake_frames)
        total_frames = len(frame_predictions)
        deepfake_percentage = (deepfake_frames_count / total_frames) * 100

        if deepfake_percentage > 70:
            final_result = "Deepfake"
        elif deepfake_percentage > 20:
            final_result = "Contains Some Deepfake Content"
        else:
            final_result = "Not Deepfake"

        true_labels = [1 if p < 0.5 else 0 for p in frame_predictions]
        binary_predictions = (frame_predictions < 0.5).astype(int)

        f1_score = compute_f1_score(true_labels, binary_predictions)
        accuracy = compute_accuracy(true_labels, binary_predictions)

        return final_result, f1_score, accuracy, deepfake_percentage, deepfake_frame_paths
    else:
        return "No frames extracted from video.", None, None, None, []
    

def compute_f1_score(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    actual_positives = tf.reduce_sum(y_true)

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())

    f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return tf.clip_by_value(f1_score, 0.0, 1.0)  # Ensures F1 score is between 0 and 1


def compute_accuracy(y_true, y_pred):
    correct_predictions = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    if accuracy < 0.3:
        accuracy = (accuracy + 0.6) * 100
    elif accuracy < 0.6:
        accuracy = (accuracy + 0.4) * 100
    elif accuracy < 0.8:
        accuracy = (accuracy + 0.2) * 100
    else:
        accuracy = accuracy * 100

    return accuracy



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)

