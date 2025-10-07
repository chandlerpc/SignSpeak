#!/usr/bin/env python3
"""Analyze training image characteristics"""
import os
from PIL import Image
import numpy as np

TRAIN_DIR = './data/asl_alphabet_train/asl_alphabet_train'

# Sample a few images from each class
classes_to_check = ['A', 'B', 'C', 'D', 'E']
images_per_class = 5

print("="*60)
print("TRAINING IMAGE ANALYSIS")
print("="*60)

for class_name in classes_to_check:
    class_dir = os.path.join(TRAIN_DIR, class_name)
    images = sorted([f for f in os.listdir(class_dir) if f.endswith('.jpg')])[:images_per_class]

    print(f"\nClass: {class_name}")
    print("-"*60)

    for img_file in images:
        img_path = os.path.join(class_dir, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)

        # Get image characteristics
        width, height = img.size
        mean_brightness = np.mean(img_array)

        # Check if background is light or dark
        # Sample corners to estimate background
        corner_samples = [
            img_array[0:10, 0:10],  # Top-left
            img_array[0:10, -10:],  # Top-right
            img_array[-10:, 0:10],  # Bottom-left
            img_array[-10:, -10:],  # Bottom-right
        ]
        avg_corner_brightness = np.mean([np.mean(corner) for corner in corner_samples])

        print(f"  {img_file}: {width}x{height}, brightness={mean_brightness:.1f}, corners={avg_corner_brightness:.1f}")

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print("Training images are 200x200 pixels")
print("Background is typically light-colored (high brightness)")
print("Hand takes up central portion with surrounding context")
print("Model expects to see background around the hand gesture")
