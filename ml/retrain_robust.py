"""Retrain with AGGRESSIVE augmentation to make model robust to background/lighting changes"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import base64
import io
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 50

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

class RobustASLDataGenerator(keras.utils.Sequence):
    """Data generator with AGGRESSIVE augmentation to prevent background overfitting"""

    def __init__(self, base64_images, labels, batch_size, shuffle=True, augment=False):
        self.base64_images = base64_images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.base64_images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.base64_images) / self.batch_size))

    def aggressive_augment(self, img):
        """Apply VERY aggressive augmentation to break background dependency"""
        # Random brightness (±40%)
        brightness_factor = np.random.uniform(0.6, 1.4)
        img = np.clip(img * brightness_factor, 0, 1)

        # Random contrast
        contrast_factor = np.random.uniform(0.7, 1.3)
        mean = img.mean()
        img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

        # Random color shift
        color_shift = np.random.uniform(-0.2, 0.2, size=3)
        img = np.clip(img + color_shift, 0, 1)

        # Random blur (simulates different camera quality)
        if np.random.random() < 0.3:
            from scipy import ndimage
            sigma = np.random.uniform(0.5, 1.5)
            img = ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))

        return img

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for idx in batch_indexes:
            try:
                img = decode_base64_image(self.base64_images[idx])

                # Apply aggressive augmentation
                if self.augment:
                    img = self.aggressive_augment(img)

                batch_images.append(img)
                batch_labels.append(self.labels[idx])
            except Exception as e:
                print(f"Error decoding image at index {idx}: {e}")
                continue

        X = np.array(batch_images)
        y = np.array(batch_labels)

        # Additional Keras augmentation
        if self.augment and len(X) > 0:
            augmentation = keras.Sequential([
                layers.RandomRotation(0.1),  # ±10 degrees
                layers.RandomZoom(0.2),      # ±20% zoom
                layers.RandomTranslation(0.1, 0.1),  # ±10% translation
            ])
            X = augmentation(X, training=True)

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

def create_model(num_classes):
    """Create MobileNetV2-based model"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Increased dropout
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    print("="*80)
    print("ROBUST TRAINING WITH AGGRESSIVE AUGMENTATION")
    print("Goal: Make model focus on HANDS, not background")
    print("="*80)

    # Load data
    json_path = './training_data_deduplicated_500.json'
    base64_images, labels, class_names = load_json_data_lazy(json_path)

    # Split into train/validation
    print("\nSplitting data...")
    train_idx, val_idx = train_test_split(
        np.arange(len(base64_images)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Create generators
    print("Creating data generators with AGGRESSIVE augmentation...")
    train_images = [base64_images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [base64_images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_generator = RobustASLDataGenerator(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True  # AGGRESSIVE augmentation enabled
    )

    val_generator = RobustASLDataGenerator(
        val_images, val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    print(f"Train batches: {len(train_generator)}")
    print(f"Validation batches: {len(val_generator)}")

    # Create model
    print("\nBuilding model...")
    model = create_model(len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    os.makedirs('./checkpoints', exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            './checkpoints/robust_mobilenet_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.CSVLogger('./training_robust_augmented_history.csv')
    ]

    # Train Phase 1: Transfer learning
    print("\n" + "="*80)
    print("TRAINING PHASE 1: Transfer Learning (frozen base)")
    print("="*80)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save('./checkpoints/robust_mobilenet_final.keras')
    print(f"\nModel saved to ./checkpoints/robust_mobilenet_final.keras")

    # Save for inference
    model.save('./checkpoints/sequential_inference.h5')
    print(f"Inference model saved to ./checkpoints/sequential_inference.h5")

    # Save class labels
    labels_data = {
        'classes': class_names,
        'num_classes': len(class_names)
    }

    os.makedirs('../public/models/asl_model', exist_ok=True)
    with open('../public/models/asl_model/class_labels.json', 'w') as f:
        json.dump(labels_data, f, indent=2)

    print("Class labels saved!")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("Model should now be robust to:")
    print("  - Different backgrounds")
    print("  - Different lighting conditions")
    print("  - Different camera quality")
    print("  - Position variations")

if __name__ == '__main__':
    main()