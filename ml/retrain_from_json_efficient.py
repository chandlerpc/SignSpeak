"""Memory-efficient training - uses data generator to load images in batches"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import base64
import io
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 160
BATCH_SIZE = 32  # Increased since we're not loading all into memory
EPOCHS = 100

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

    def __init__(self, base64_images, labels, batch_size, shuffle=True, augment=False):
        self.base64_images = base64_images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.base64_images))

        # Data augmentation layers
        if augment:
            self.augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.15),
                layers.RandomBrightness(0.1),
            ])

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.base64_images) / self.batch_size))

    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Decode images in this batch
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

        # Apply augmentation if enabled
        if self.augment and len(X) > 0:
            X = self.augmentation(X, training=True)

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
        print(f"Loading {letter}: {len(letter_images)} images...")
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
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrain_from_json_efficient.py <path_to_json_file>")
        sys.exit(1)

    json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print("="*80)
    print("MEMORY-EFFICIENT TRAINING WITH DATA GENERATOR")
    print("="*80)

    # Load data (base64 strings only, not decoded)
    base64_images, labels, class_names = load_json_data_lazy(json_path)

    # Split into train/validation indices
    print("\nSplitting data...")
    train_idx, val_idx = train_test_split(
        np.arange(len(base64_images)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Create generators
    print("Creating data generators...")
    train_images = [base64_images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [base64_images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_generator = ASLDataGenerator(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )

    val_generator = ASLDataGenerator(
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
            './checkpoints/mobilenet_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
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
        keras.callbacks.CSVLogger('./training_history.csv')
    ]

    # Train
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

    # Fine-tune
    print("\n" + "="*80)
    print("TRAINING PHASE 2: Fine-tuning (unfrozen base)")
    print("="*80)

    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks[1].patience = 10  # Reduce patience for fine-tuning

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    val_loss, val_acc = model.evaluate(val_generator, verbose=1)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

    # Save model
    model.save('./checkpoints/mobilenet_final.keras')
    print(f"Model saved to ./checkpoints/mobilenet_final.keras")

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
    print(f"Best model: ./checkpoints/mobilenet_best.keras")
    print(f"Final model: ./checkpoints/mobilenet_final.keras")
    print(f"Final accuracy: {val_acc*100:.2f}%")

if __name__ == '__main__':
    main()