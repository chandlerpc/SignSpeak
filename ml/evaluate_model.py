"""Evaluate trained model with detailed metrics"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import base64
import io
import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 160
BATCH_SIZE = 32

def decode_base64_image(base64_str):
    """Decode base64 string to numpy array"""
    if 'data:image' in base64_str:
        base64_str = base64_str.split(',')[1]

    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array

class ASLDataGenerator(keras.utils.Sequence):
    """Custom data generator that loads images on-the-fly"""

    def __init__(self, base64_images, labels, batch_size, shuffle=False):
        self.base64_images = base64_images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.base64_images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.base64_images) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for idx in batch_indexes:
            try:
                img = decode_base64_image(self.base64_images[idx])
                batch_images.append(img)
                batch_labels.append(self.labels[idx])
            except Exception as e:
                print(f"Error decoding image at index {idx}: {e}")
                continue

        X = np.array(batch_images)
        y = np.array(batch_labels)

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def load_json_data_lazy(json_path):
    """Load JSON but keep base64 strings (don't decode all images)"""
    print(f"Loading data from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    base64_images = []
    labels = []
    class_names = sorted(data.keys())

    print(f"Found {len(class_names)} classes: {class_names}")

    for class_idx, letter in enumerate(class_names):
        letter_images = data[letter]
        print(f"  {letter}: {len(letter_images)} images")
        base64_images.extend(letter_images)
        labels.extend([class_idx] * len(letter_images))

    print(f"Total: {len(base64_images)} images")
    return base64_images, labels, class_names

def evaluate_model(model_path, data_path, use_validation_split=True):
    """
    Evaluate model with detailed metrics

    Args:
        model_path: Path to trained .keras model
        data_path: Path to JSON data file
        use_validation_split: If True, use same validation split as training (20%, seed 42)
    """
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print(f"Model loaded!")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Load data
    base64_images, labels, class_names = load_json_data_lazy(data_path)

    if use_validation_split:
        # Use same split as training (80/20, seed 42)
        print("\nUsing validation split from training (20% of data, seed=42)...")
        train_idx, test_idx = train_test_split(
            np.arange(len(base64_images)),
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        test_images = [base64_images[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
    else:
        # Use all data
        print("\nUsing all data for evaluation...")
        test_images = base64_images
        test_labels = labels

    print(f"Test set size: {len(test_images)} images")

    # Create test generator
    test_generator = ASLDataGenerator(
        test_images, test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATING MODEL...")
    print("="*80)

    loss, accuracy = model.evaluate(test_generator, verbose=1)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Get predictions for confusion matrix
    print("\nGenerating predictions for detailed analysis...")
    y_true = []
    y_pred = []

    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        predictions = model.predict(X_batch, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y_batch)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report
    print("\n" + "="*80)
    print("PER-CLASS PERFORMANCE")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, letter in enumerate(class_names):
        total = cm.sum(axis=1)[i]
        correct = cm.diagonal()[i]
        acc = per_class_accuracy[i] * 100
        print(f"{letter}: {acc:.2f}% ({correct}/{total} correct)")

    # Find worst performing classes
    print("\n" + "="*80)
    print("WORST PERFORMING CLASSES (Bottom 5)")
    print("="*80)
    worst_indices = np.argsort(per_class_accuracy)[:5]
    for idx in worst_indices:
        letter = class_names[idx]
        acc = per_class_accuracy[idx] * 100
        total = cm.sum(axis=1)[idx]
        correct = cm.diagonal()[idx]
        print(f"{letter}: {acc:.2f}% ({correct}/{total})")

        # Show what it was confused with
        confused_with = cm[idx]
        top_confusion = np.argsort(confused_with)[::-1][1:4]  # Top 3 (excluding itself)
        for conf_idx in top_confusion:
            if confused_with[conf_idx] > 0:
                print(f"  -> Confused with {class_names[conf_idx]}: {confused_with[conf_idx]} times")

    # Save confusion matrix plot
    print("\n" + "="*80)
    print("SAVING CONFUSION MATRIX PLOT")
    print("="*80)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy*100:.2f}%')
    plt.colorbar(label='Count')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plot_path = './evaluation_confusion_matrix.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {plot_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_model.py <model_path> <data_path> [--full]")
        print("\nOptions:")
        print("  --full    Evaluate on full dataset instead of validation split")
        print("\nExamples:")
        print("  python evaluate_model.py ./checkpoints/mobilenet_final.keras ./training_data_deduplicated_500.json")
        print("  python evaluate_model.py ./checkpoints/mobilenet_final.keras ./corrected_training_data.json --full")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    use_validation_split = '--full' not in sys.argv

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    evaluate_model(model_path, data_path, use_validation_split)

if __name__ == '__main__':
    main()
