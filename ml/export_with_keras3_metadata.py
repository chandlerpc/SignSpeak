"""Export model with FULL Keras 3 metadata (module, registered_name, dtype objects)
This matches the format that TensorFlow.js v4 expects
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from tensorflow import keras

def add_keras3_metadata(config):
    """Add Keras 3 metadata fields that TF.js v4 expects"""
    if isinstance(config, dict):
        # Add module and registered_name to layers
        if 'class_name' in config and config.get('class_name') in [
            'InputLayer', 'Conv2D', 'BatchNormalization', 'Activation',
            'MaxPooling2D', 'GlobalAveragePooling2D', 'Dropout', 'Dense',
            'RandomFlip', 'RandomRotation', 'RandomZoom'
        ]:
            if 'module' not in config:
                config['module'] = 'keras.layers'
            if 'registered_name' not in config:
                config['registered_name'] = None

        # Convert simple dtype strings to dtype objects
        if 'dtype' in config and isinstance(config['dtype'], str):
            config['dtype'] = {
                'module': 'keras',
                'class_name': 'DTypePolicy',
                'config': {'name': config['dtype']},
                'registered_name': None
            }

        # Recursively process nested configs
        for key, value in config.items():
            config[key] = add_keras3_metadata(value)

        return config
    elif isinstance(config, list):
        return [add_keras3_metadata(item) for item in config]
    else:
        return config

# Find model file
model_paths = [
    './checkpoints/sequential_inference.h5',
    './checkpoints/sequential_inference.keras',
    './checkpoints/custom_asl_best.keras',
    './checkpoints/model_best.h5'
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if not model_path:
    print("Error: No model file found!")
    exit(1)

print(f"Loading model: {model_path}")
model = keras.models.load_model(model_path)
print(f"Model type: {model.__class__.__name__}")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Get config
config = model.get_config()
print(f"\nOriginal config keys: {list(config.keys())}")

# Add Keras 3 metadata
print("Adding Keras 3 metadata (module, registered_name, dtype objects)...")
config_with_metadata = add_keras3_metadata(config)

# Also ensure top-level dtype is an object
if 'dtype' in config_with_metadata and isinstance(config_with_metadata['dtype'], str):
    config_with_metadata['dtype'] = {
        'module': 'keras',
        'class_name': 'DTypePolicy',
        'config': {'name': config_with_metadata['dtype']},
        'registered_name': None
    }

# Verify first layer structure
first_layer = config_with_metadata.get('layers', [{}])[0]
print(f"\nFirst layer structure:")
print(f"  - class_name: {first_layer.get('class_name')}")
print(f"  - module: {first_layer.get('module')}")
print(f"  - registered_name: {first_layer.get('registered_name')}")
print(f"  - dtype: {type(first_layer.get('config', {}).get('dtype'))}")

# Create output directory
output_dir = '../public/models/asl_sequential'
os.makedirs(output_dir, exist_ok=True)

# Extract weights
print("\nExtracting weights...")
weights_list = []
weight_specs = []

for i, layer in enumerate(model.layers):
    layer_weights = layer.get_weights()
    if len(layer_weights) > 0:
        layer_name = layer.name
        print(f"  Layer {i}: {layer_name} ({len(layer_weights)} tensors)")

        for j, w in enumerate(layer_weights):
            # Determine weight name
            if 'conv2d' in layer_name or 'conv_' in layer_name:
                weight_name = f"{layer_name}/{'kernel' if j == 0 else 'bias'}"
            elif 'batch_normalization' in layer_name or 'bn_' in layer_name:
                names = ['gamma', 'beta', 'moving_mean', 'moving_variance']
                weight_name = f"{layer_name}/{names[j] if j < len(names) else f'param_{j}'}"
            elif 'dense' in layer_name:
                weight_name = f"{layer_name}/{'kernel' if j == 0 else 'bias'}"
            else:
                weight_name = f"{layer_name}/param_{j}"

            weight_specs.append({
                "name": weight_name,
                "shape": list(w.shape),
                "dtype": "float32"
            })
            weights_list.append(w.astype(np.float32).flatten())

# Concatenate all weights
all_weights = np.concatenate(weights_list)

# Save weights binary
shard_path = 'group1-shard1of1.bin'
with open(f'{output_dir}/{shard_path}', 'wb') as f:
    f.write(all_weights.tobytes())

print(f"\nSaved {len(all_weights)} weight values ({all_weights.nbytes:,} bytes)")

# Create TensorFlow.js model.json
model_json = {
    "format": "layers-model",
    "generatedBy": "keras v3.0",
    "convertedBy": "TensorFlow.js Converter v4.20.0",
    "modelTopology": config_with_metadata,
    "weightsManifest": [{
        "paths": [shard_path],
        "weights": weight_specs
    }]
}

# Save model.json
model_json_path = f'{output_dir}/model.json'
with open(model_json_path, 'w') as f:
    json.dump(model_json, f, indent=2)

print(f"\nModel JSON saved: {model_json_path}")
print(f"Total size: {os.path.getsize(model_json_path) + all_weights.nbytes:,} bytes")

# Copy class labels
if os.path.exists('./checkpoints/class_labels.json'):
    import shutil
    shutil.copy('./checkpoints/class_labels.json', f'{output_dir}/class_labels.json')
    print("Copied class_labels.json")

print("\n" + "="*60)
print("SUCCESS! Model exported with Keras 3 metadata for TF.js v4")
print("="*60)
print(f"Output: {output_dir}")
print("Format: FULL Keras 3 format (module, registered_name, dtype objects)")
