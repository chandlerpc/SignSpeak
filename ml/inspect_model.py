#!/usr/bin/env python3
"""Inspect the trained model architecture"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

model_path = './checkpoints/simple_model_best.h5'

print(f"Loading model from {model_path}...")
model = keras.models.load_model(model_path)

print(f"\n{'='*60}")
print("MODEL ARCHITECTURE")
print('='*60)
print(f"\nInput shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

print(f"\n{'='*60}")
print("LAYER SUMMARY")
print('='*60)
model.summary()

print(f"\n{'='*60}")
print("LAYER COUNT")
print('='*60)
print(f"Total layers: {len(model.layers)}")

print(f"\n{'='*60}")
print("DETAILED LAYER INFORMATION")
print('='*60)
for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}: {layer.name}")
    print(f"  Type: {type(layer).__name__}")
    print(f"  Output shape: {layer.output_shape}")
    if hasattr(layer, 'trainable'):
        print(f"  Trainable: {layer.trainable}")
    if hasattr(layer, 'count_params'):
        print(f"  Parameters: {layer.count_params():,}")

print(f"\n{'='*60}")
print("TOTAL PARAMETERS")
print('='*60)
total_params = model.count_params()
trainable_params = sum([layer.count_params() for layer in model.layers if hasattr(layer, 'trainable') and layer.trainable])
non_trainable_params = total_params - trainable_params

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
