"""Simple Flask server to serve model predictions"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load model at startup
print("Loading sequential inference model...")
model = keras.models.load_model('./checkpoints/sequential_inference.h5')
print(f"Model loaded! Input: {model.input_shape}, Output: {model.output_shape}")
print("Model: Sequential inference model (160x160 input, 26 classes)")

# Load class labels
with open('../public/models/asl_model/class_labels.json', 'r') as f:
    labels_data = json.load(f)
    class_labels = labels_data['classes']

print(f"Loaded {len(class_labels)} class labels")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = np.array(data['image'], dtype=np.float32)

        # Ensure shape is (1, 160, 160, 3) for sequential model
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)

        # Verify correct input shape
        expected_shape = (1, 160, 160, 3)
        if image_data.shape != expected_shape:
            return jsonify({'error': f'Invalid input shape. Expected {expected_shape}, got {image_data.shape}'}), 400

        # Make prediction
        predictions = model.predict(image_data, verbose=0)

        # Get top prediction
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])

        if predicted_idx >= len(class_labels):
            predicted_class = f"UNKNOWN_CLASS_{predicted_idx}"
        else:
            predicted_class = class_labels[predicted_idx]

        # Log prediction for debugging
        print(f">>> Prediction: {predicted_class} (confidence: {confidence:.2%})", flush=True)

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    print("\nModel server ready on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
