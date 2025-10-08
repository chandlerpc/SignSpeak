"""Train ASL model from JSON data collected via DataCollector component"""
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
BATCH_SIZE = 16
EPOCHS = 500

def decode_base64_image(base64_str):
    """Decode base64 string to numpy array"""
    # Remove data URL prefix if present
    if 'data:image' in base64_str:
        base64_str = base64_str.split(',')[1]

    # Decode base64 to image
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('RGB')  # Ensure RGB format
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
    class_names = sorted(data.keys())  # Alphabetical order

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
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrain_from_json.py <path_to_json_file>")
        print("Example: python retrain_from_json.py ../training_data/asl_training_data_1234567890.json")
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

    print(f"Train set: {len(X_train)} images")
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

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            './checkpoints/custom_model_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=50,
            restore_best_weights=False,  # Avoid Python 3.13 bug
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ])

    # Augment training data
    print("\nApplying data augmentation...")
    X_train_aug = data_augmentation(X_train, training=True)

    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    history = model.fit(
        X_train_aug, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tune (unfreeze base model)
    print("\n" + "="*50)
    print("Fine-tuning with unfrozen base model...")
    print("="*50 + "\n")

    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        X_train_aug, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

    # Save final model
    model.save('./checkpoints/custom_model_final.h5')
    print(f"\nModel saved to ./checkpoints/custom_model_final.h5")

    # Save class labels
    labels_data = {
        'classes': class_names,
        'num_classes': len(class_names)
    }

    with open('../public/models/asl_model/class_labels.json', 'w') as f:
        json.dump(labels_data, f, indent=2)

    print(f"Class labels saved to ../public/models/asl_model/class_labels.json")

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Copy ./checkpoints/custom_model_final.h5 to ./checkpoints/model_best.h5")
    print("2. Restart the model server: python model_server.py")
    print("3. Test the new model in the application")

if __name__ == '__main__':
    main()