"""
Data Preprocessing and Loading Module
Handles image loading, resizing (224x224), data augmentation, normalization, and generators.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

def get_data_generators(dataset_dir="dataset", target_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Creates train, validation, and test data generators with augmentation and normalization.
    
    Args:
        dataset_dir (str): Base directory containing train, val, and test folders.
        target_size (tuple): Target image dimensions (height, width).
        batch_size (int): Number of images per batch.
        
    Returns:
        tuple: (train_generator, val_generator, test_generator, class_labels)
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    # Data Augmentation & Normalization for Training Set
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest"
    )

    # Only Normalization (Rescaling) for Validation and Test Sets
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    print(f"Loading training data from {train_dir}...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    print(f"Loading validation data from {val_dir}...")
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    print(f"Loading test data from {test_dir}...")
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # Extract class labels dictionary and sorted list
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    print(f"Detected Classes: {class_indices}")

    return train_generator, val_generator, test_generator, class_labels

if __name__ == "__main__":
    train_gen, val_gen, test_gen, labels = get_data_generators()
    print("Data generators successfully initialized.")
    print("Class mapping:", labels)
