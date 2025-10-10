"""Debug script to check what the training data looks like"""
import json
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load a few samples from training data
with open('./training_data_deduplicated_500.json', 'r') as f:
    data = json.load(f)

# Get first image from letter 'B'
letter = 'B'
base64_str = data[letter][0]

# Decode
if 'data:image' in base64_str:
    base64_str = base64_str.split(',')[1]

img_data = base64.b64decode(base64_str)
img = Image.open(io.BytesIO(img_data))

print(f"Original image from training data:")
print(f"  Size: {img.size}")
print(f"  Mode: {img.mode}")
print(f"  Format: {img.format}")

# Show what happens after resize
img_resized = img.convert('RGB').resize((160, 160))
img_array = np.array(img_resized, dtype=np.float32) / 255.0

print(f"\nAfter preprocessing:")
print(f"  Shape: {img_array.shape}")
print(f"  Min: {img_array.min():.4f}")
print(f"  Max: {img_array.max():.4f}")
print(f"  Mean: {img_array.mean():.4f}")

# Save sample
plt.figure(figsize=(6, 6))
plt.imshow(img_resized)
plt.title(f"Training Data Sample - Letter {letter}")
plt.axis('off')
plt.savefig('./training_data_sample.png')
print(f"\nSaved sample to training_data_sample.png")

# Check a few more samples
print(f"\n\nChecking 5 samples from letter {letter}:")
for i in range(min(5, len(data[letter]))):
    base64_str = data[letter][i]
    if 'data:image' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    print(f"  Sample {i}: {img.size} - {img.mode}")