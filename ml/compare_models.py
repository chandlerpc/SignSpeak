#!/usr/bin/env python3
"""Compare performance of old vs new model on test images"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from PIL import Image
import json
import glob

# Paths
OLD_MODEL = './checkpoints/simple_model_best.h5'
NEW_MODEL = './checkpoints/asl_alphabet_best.h5'
TEST_DIR = './data/asl_alphabet_test/asl_alphabet_test'
LABELS_FILE = '../public/models/asl_model/class_labels.json'

# Image size for each model
OLD_IMG_SIZE = 64
NEW_IMG_SIZE = 64

def load_image(image_path, img_size):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_true_label(filename):
    """Extract true label from filename (e.g., A_test.jpg -> A)"""
    return filename.split('_test')[0]

def main():
    print("="*70)
    print("MODEL COMPARISON ON TEST IMAGES")
    print("="*70)

    # Load class labels
    with open(LABELS_FILE, 'r') as f:
        labels_data = json.load(f)
        class_labels = labels_data['classes']

    print(f"\nClasses: {class_labels}")
    print(f"Total classes: {len(class_labels)}")

    # Load models
    print("\nLoading models...")
    print(f"  Old model: {OLD_MODEL}")
    old_model = keras.models.load_model(OLD_MODEL)
    print(f"  New model: {NEW_MODEL}")
    new_model = keras.models.load_model(NEW_MODEL)

    # Get test images
    test_images = sorted(glob.glob(os.path.join(TEST_DIR, '*_test.jpg')))
    print(f"\nFound {len(test_images)} test images")

    # Compare predictions
    print("\n" + "="*70)
    print("PREDICTIONS COMPARISON")
    print("="*70)
    print(f"{'Image':<20} {'True':<10} {'Old Model':<20} {'New Model':<20} {'Winner':<10}")
    print("-"*70)

    old_correct = 0
    new_correct = 0
    both_correct = 0
    both_wrong = 0

    results = []

    for img_path in test_images:
        filename = os.path.basename(img_path)
        true_label = get_true_label(filename)

        # Load image for each model
        old_img = load_image(img_path, OLD_IMG_SIZE)
        new_img = load_image(img_path, NEW_IMG_SIZE)

        # Get predictions
        old_pred = old_model.predict(old_img, verbose=0)
        new_pred = new_model.predict(new_img, verbose=0)

        old_idx = int(np.argmax(old_pred[0]))
        new_idx = int(np.argmax(new_pred[0]))

        old_conf = float(old_pred[0][old_idx])
        new_conf = float(new_pred[0][new_idx])

        old_label = class_labels[old_idx] if old_idx < len(class_labels) else f"UNKNOWN_{old_idx}"
        new_label = class_labels[new_idx] if new_idx < len(class_labels) else f"UNKNOWN_{new_idx}"

        # Check correctness
        old_is_correct = (old_label.upper() == true_label.upper())
        new_is_correct = (new_label.upper() == true_label.upper())

        if old_is_correct:
            old_correct += 1
        if new_is_correct:
            new_correct += 1
        if old_is_correct and new_is_correct:
            both_correct += 1
        if not old_is_correct and not new_is_correct:
            both_wrong += 1

        # Determine winner
        if old_is_correct and not new_is_correct:
            winner = "OLD"
        elif new_is_correct and not old_is_correct:
            winner = "NEW"
        elif old_is_correct and new_is_correct:
            winner = "BOTH OK"
        else:
            winner = "BOTH NO"

        # Format output
        old_result = f"{old_label} ({old_conf:.1%})"
        new_result = f"{new_label} ({new_conf:.1%})"

        # Add indicators
        if old_is_correct:
            old_result = "[OK] " + old_result
        else:
            old_result = "[NO] " + old_result

        if new_is_correct:
            new_result = "[OK] " + new_result
        else:
            new_result = "[NO] " + new_result

        print(f"{filename:<20} {true_label:<10} {old_result:<20} {new_result:<20} {winner:<10}")

        results.append({
            'image': filename,
            'true_label': true_label,
            'old_prediction': old_label,
            'old_confidence': old_conf,
            'old_correct': old_is_correct,
            'new_prediction': new_label,
            'new_confidence': new_conf,
            'new_correct': new_is_correct
        })

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_images = len(test_images)
    old_accuracy = (old_correct / total_images) * 100
    new_accuracy = (new_correct / total_images) * 100
    improvement = new_accuracy - old_accuracy

    print(f"\nTotal test images: {total_images}")
    print(f"\nOld Model (simple_model_best.h5):")
    print(f"  Correct: {old_correct}/{total_images}")
    print(f"  Accuracy: {old_accuracy:.2f}%")

    print(f"\nNew Model (asl_alphabet_best.h5):")
    print(f"  Correct: {new_correct}/{total_images}")
    print(f"  Accuracy: {new_accuracy:.2f}%")

    print(f"\nImprovement: {improvement:+.2f}%")

    print(f"\nBoth models correct: {both_correct}/{total_images} ({both_correct/total_images*100:.1f}%)")
    print(f"Both models wrong: {both_wrong}/{total_images} ({both_wrong/total_images*100:.1f}%)")
    print(f"Only old model correct: {old_correct - both_correct}/{total_images}")
    print(f"Only new model correct: {new_correct - both_correct}/{total_images}")

    # Determine winner
    print("\n" + "="*70)
    if new_accuracy > old_accuracy:
        print(f"WINNER: New Model (+{improvement:.2f}% improvement)")
    elif old_accuracy > new_accuracy:
        print(f"WINNER: Old Model (+{-improvement:.2f}% better)")
    else:
        print("TIE: Both models performed equally")
    print("="*70)

if __name__ == '__main__':
    main()
