"""Create a proper Sequential model for inference (no augmentation, no Functional API)"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np

print("Loading trained model...")
trained_model = keras.models.load_model('./checkpoints/custom_asl_best.h5')

print("\nOriginal model layers:")
for i, layer in enumerate(trained_model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")

# Create new Sequential model WITHOUT data augmentation layers
print("\nBuilding Sequential inference model...")
inference_model = keras.Sequential(name='asl_sequential_inference')

# Skip augmentation layers (0-2) and add all actual processing layers (3+)
for i, layer in enumerate(trained_model.layers[3:], start=3):
    print(f"Adding layer {i}: {layer.name}")

    # Get layer config and weights
    config = layer.get_config()
    weights = layer.get_weights()

    # Recreate layer from config
    layer_class = layer.__class__
    new_layer = layer_class.from_config(config)

    # Add to model
    if i == 3:  # First layer - specify input shape
        inference_model.add(keras.layers.Input(shape=(128, 128, 3)))
        inference_model.add(new_layer)
    else:
        inference_model.add(new_layer)

    # Set weights
    if len(weights) > 0:
        new_layer.set_weights(weights)

print("\nInference model summary:")
inference_model.summary()

# Test the model
print("\nTesting inference model...")
test_input = np.random.rand(1, 128, 128, 3).astype('float32')
output = inference_model.predict(test_input, verbose=0)
print(f"Output shape: {output.shape}")
print(f"Sum of probabilities: {output.sum():.4f}")

# Save as H5
save_path = './checkpoints/sequential_inference.h5'
inference_model.save(save_path)
print(f"\nSaved Sequential inference model: {save_path}")
print(f"Model type: {type(inference_model).__name__}")
print(f"Model class: {inference_model.__class__.__name__}")
