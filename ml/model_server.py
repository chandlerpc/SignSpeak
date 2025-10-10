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
print("Loading simple model (simple_final.keras)...")
model = keras.models.load_model('./checkpoints/simple_final.keras')
print(f"Model loaded! Input: {model.input_shape}, Output: {model.output_shape}")
print("Model: Simple MobileNetV2 (128x128 input, no crop, direct resize)")

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

        # Ensure shape is (1, 128, 128, 3) for hand-focused model
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)

        # Verify correct input shape
        expected_shape = (1, 128, 128, 3)
        if image_data.shape != expected_shape:
            return jsonify({'error': f'Invalid input shape. Expected {expected_shape}, got {image_data.shape}'}), 400

        # SIMPLE preprocessing - just like training: already normalized to [0,1]
        # No extra brightness normalization - keep it simple!
        print(f"Image stats - Min: {image_data.min():.4f}, Max: {image_data.max():.4f}, Mean: {image_data.mean():.4f}")

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
