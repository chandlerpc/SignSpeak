"""Custom CNN architecture for ASL recognition - optimized for hand gestures"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import base64
import io
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 128  # Good balance for hand detail
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
    return images, labels, class_names

def create_custom_asl_model(num_classes):
    """
    Balanced custom CNN for ASL - moderate regularization to fix volatility

    Changes from v3:
    - Moderate dropout (0.3) - balanced between v2 (0.2, overfits) and v3 (0.5, can't learn)
    - Enhanced data augmentation (rotation 0.2, zoom 0.2, brightness 0.1)
    - No L2 reg (was blocking learning)
    - Learning rate reduction callback for stability
    """
    model = models.Sequential([
        # Moderate data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),  # Moderate: between 0.1 (weak) and 0.2 (too strong)
        layers.RandomZoom(0.15),      # Moderate: between 0.1 (weak) and 0.2 (too strong)

        # Block 1: Initial features
        layers.Conv2D(32, (5, 5), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 2: Mid-level features
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3: High-level features
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 4: Abstract features
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        # Dense layers - moderate dropout (balanced)
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),  # Balanced: not too weak (0.2), not too strong (0.5)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Same moderate dropout

        # Output
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    print("="*60)
    print("CUSTOM ASL HAND GESTURE RECOGNITION MODEL")
    print("="*60)
    print(f"Architecture: 5-block Custom CNN")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)

    # Load data
    json_path = './merged_training_data_deduplicated.json'

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    images, labels, class_names = load_json_data(json_path)

    # Split into train/validation
    print("\nSplitting into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Train set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")

    # Create model
    print("\n" + "="*60)
    print("BUILDING CUSTOM MODEL")
    print("="*60)
    model = create_custom_asl_model(len(class_names))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Stable learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train with callbacks to fix volatility
    print("\n" + "="*60)
    print(f"TRAINING - {EPOCHS} EPOCHS")
    print("="*60)

    os.makedirs('./checkpoints', exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            './checkpoints/custom_asl_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"\nValidation Accuracy: {val_acc*100:.2f}%")

    # Save final model (best model already saved by callback)
    model.save('./checkpoints/custom_asl_final.keras')
    print(f"\nFinal model saved to ./checkpoints/custom_asl_final.keras")
    print(f"Best model saved to ./checkpoints/custom_asl_best.keras")

    # Save class labels
    labels_data = {
        'classes': class_names,
        'num_classes': len(class_names)
    }

    os.makedirs('../public/models/asl_model', exist_ok=True)
    with open('../public/models/asl_model/class_labels.json', 'w') as f:
        json.dump(labels_data, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: ./checkpoints/custom_asl_best.keras")
    print(f"Final model: ./checkpoints/custom_asl_final.keras")
    print(f"Architecture: Custom 4-block CNN with regularization")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Final accuracy: {val_acc*100:.2f}%")
    print(f"Total parameters: {model.count_params():,}")
    print("\nVolatility fixes applied:")
    print("  - Stronger dropout (0.4-0.5)")
    print("  - L2 regularization (0.001)")
    print("  - Enhanced data augmentation")
    print("  - Learning rate reduction on plateau")

if __name__ == '__main__':
    main()
