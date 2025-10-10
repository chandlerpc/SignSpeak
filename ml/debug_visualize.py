"""Visualize training samples to verify labels are correct"""
import json
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load training data
with open('training_data_deduplicated_500.json', 'r') as f:
    data = json.load(f)

# Get class names in sorted order (same as training)
class_names = sorted(data.keys())

# Visualize first 5 samples from first 6 letters (A, B, C, D, E, F)
fig, axes = plt.subplots(6, 5, figsize=(15, 18))
fig.suptitle('Training Data Visualization (First 5 samples of A-F)', fontsize=16)

for class_idx, letter in enumerate(class_names[:6]):
    samples = data[letter]
    for sample_idx in range(5):
        base64_str = samples[sample_idx]

        # Decode exactly like training script
        if 'data:image' in base64_str:
            base64_str = base64_str.split(',')[1]

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')

        # Show original 160x160 image (as stored)
        ax = axes[class_idx, sample_idx]
        ax.imshow(img)
        ax.set_title(f'{letter} #{sample_idx+1}')
        ax.axis('off')

plt.tight_layout()
plt.savefig('training_samples_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to training_samples_visualization.png")
print("\nPlease verify:")
print("- Row 1 (A): Should show closed fist with thumb on side")
print("- Row 2 (B): Should show flat hand, fingers together")
print("- Row 3 (C): Should show curved hand forming C shape")
print("- Row 4 (D): Should show index finger up, other fingers on thumb")
print("- Row 5 (E): Should show fingers curled over thumb")
print("- Row 6 (F): Should show OK sign (thumb and index making circle)")