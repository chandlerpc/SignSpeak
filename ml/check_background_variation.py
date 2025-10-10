"""Check if training data has background variation"""
import json
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load training data
with open('./training_data_deduplicated_500.json', 'r') as f:
    data = json.load(f)

# Sample 5 images from different letters
letters = ['A', 'F', 'K', 'P', 'Z']
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for col, letter in enumerate(letters):
    for row in range(2):
        idx = row * 50  # Sample from different parts of the dataset
        base64_str = data[letter][idx]

        if 'data:image' in base64_str:
            base64_str = base64_str.split(',')[1]

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))

        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'{letter} #{idx}')

plt.tight_layout()
plt.savefig('./training_background_samples.png', dpi=150, bbox_inches='tight')
print("Saved training background samples to training_background_samples.png")

# Calculate background consistency (check corners which are usually background)
print("\n" + "="*60)
print("BACKGROUND CONSISTENCY ANALYSIS")
print("="*60)

def check_corners(img_array):
    """Check the 4 corners of the image (typically background)"""
    h, w = img_array.shape[:2]
    corner_size = 20

    top_left = img_array[:corner_size, :corner_size].mean()
    top_right = img_array[:corner_size, -corner_size:].mean()
    bottom_left = img_array[-corner_size:, :corner_size].mean()
    bottom_right = img_array[-corner_size:, -corner_size:].mean()

    return (top_left + top_right + bottom_left + bottom_right) / 4

corner_means = []
for letter in letters:
    for i in range(min(10, len(data[letter]))):
        base64_str = data[letter][i]
        if 'data:image' in base64_str:
            base64_str = base64_str.split(',')[1]

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_array = np.array(img)

        corner_mean = check_corners(img_array)
        corner_means.append(corner_mean)

corner_means = np.array(corner_means)
print(f"\nBackground (corners) brightness:")
print(f"  Mean: {corner_means.mean():.2f}")
print(f"  Std:  {corner_means.std():.2f}")
print(f"  Min:  {corner_means.min():.2f}")
print(f"  Max:  {corner_means.max():.2f}")

if corner_means.std() < 20:
    print("\n⚠️  WARNING: Very consistent background detected!")
    print("   Model may have learned to rely on background features.")
else:
    print("\n✓ Background has good variation.")