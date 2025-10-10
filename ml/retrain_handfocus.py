"""Train ASL model focused ONLY on hand shapes - remove background bias"""
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
import tensorflow as tf

# Configuration
IMG_SIZE = 128  # Smaller size to focus on details
BATCH_SIZE = 32
EPOCHS = 100

def decode_base64_image(base64_str):
    """Decode base64 string to numpy array with tight crop"""
    # Remove data URL prefix if present
    if 'data:image' in base64_str:
        base64_str = base64_str.split(',')[1]

    # Decode base64 to image
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('RGB')

    # TIGHT CENTER CROP - remove outer 20 pixels on each side (160 -> 120)
    # This removes background and focuses on hand
    width, height = img.size
    crop_amount = 20
    img = img.crop((crop_amount, crop_amount, width-crop_amount, height-crop_amount))

    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array

def load_json_data(json_path):
    """Load and process JSON training data"""
    print(f"Loading data from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    images = []
    labels = []
    class_names = sorted(data.keys())

    print(f"Found {len(class_names)} classes: {class_names}")

    for class_idx, letter in enumerate(class_names):
        print(f"Processing {letter}: {len(data[letter])} images...")
        for base64_img in data[letter]:
            try:
                img_array = decode_base64_image(base64_img)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error decoding image for {letter}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images total")
    print(f"Image shape: {images.shape}")
    return images, labels, class_names

def create_model(num_classes):
    """Create model with heavy dropout to prevent background overfitting"""
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.75  # Smaller model to focus on important features
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),  # Increased dropout
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),  # Increased dropout
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrain_handfocus.py <path_to_json_file>")
        sys.exit(1)

    json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Load data
    images, labels, class_names = load_json_data(json_path)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTrain set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # Create model
    model = create_model(len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    # Create checkpoint directory
    os.makedirs('./checkpoints', exist_ok=True)

    # Callbacks - monitor TRAINING accuracy to ensure model is learning
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            './checkpoints/handfocus_best.keras',
            save_best_only=True,
            monitor='accuracy',  # Monitor training accuracy!
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='accuracy',
            patience=30,
            min_delta=0.01,  # Must improve by at least 1%
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.CSVLogger('./training_handfocus_history.csv')
    ]

    # AGGRESSIVE data augmentation to force focus on hand shapes
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),  # 30% rotation
        layers.RandomZoom(0.3),  # 30% zoom
        layers.RandomTranslation(0.2, 0.2),  # Shift position
        layers.RandomBrightness(0.3),  # Vary brightness heavily
        layers.RandomContrast(0.3),  # Vary contrast
    ])

    # Train model
    print("\n" + "="*50)
    print("Training with HAND-FOCUSED preprocessing")
    print("- Tight crop (removed 20px border)")
    print("- Aggressive augmentation")
    print("- Heavy dropout (0.5)")
    print("- Monitoring TRAINING accuracy to ensure learning")
    print("="*50 + "\n")

    history = model.fit(
        data_augmentation(X_train, training=True),
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tune with unfrozen base
    print("\n" + "="*50)
    print("Fine-tuning...")
    print("="*50 + "\n")

    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        data_augmentation(X_train, training=True),
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Final evaluation
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Training Accuracy:   {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Accuracy Gap:        {abs(train_acc - val_acc)*100:.2f}%")
    print(f"{'='*50}")

    if train_acc < 0.70:
        print("\n⚠️  WARNING: Training accuracy is LOW (<70%)")
        print("   Model is NOT learning hand shapes properly!")
        print("   This data may require different people/backgrounds.")
    elif abs(train_acc - val_acc) > 0.20:
        print("\n⚠️  WARNING: Large gap between train/val accuracy")
        print("   Model may be overfitting to environment.")
    else:
        print("\n✓ Model appears to be learning hand shapes!")

    # Save final model
    model.save('./checkpoints/handfocus_final.keras')
    print(f"\nModel saved to ./checkpoints/handfocus_final.keras")

    # Save class labels
    labels_data = {
        'classes': class_names,
        'num_classes': len(class_names)
    }

    with open('../public/models/asl_model/class_labels.json', 'w') as f:
        json.dump(labels_data, f, indent=2)

    print(f"Class labels saved to ../public/models/asl_model/class_labels.json")
    print("\nRestart the model server to load the new model!")

if __name__ == '__main__':
    main()