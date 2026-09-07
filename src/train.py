"""
RiceCare AI Enterprise Model Training Script
Trains MobileNetV2 with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
Saves ricecare_ai_model.keras and ricecare_ai_model.h5
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("src"))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_loader import get_data_generators
from model import build_ricecare_model

def train_ricecare_model(
    dataset_dir="dataset",
    models_dir="models",
    assets_dir="assets",
    epochs=25,
    batch_size=16,
    learning_rate=0.0004
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Load Data
    train_gen, val_gen, test_gen, class_labels = get_data_generators(
        dataset_dir=dataset_dir,
        batch_size=batch_size
    )

    # Save class label index
    class_indices_path = os.path.join(models_dir, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(class_labels, f, indent=4)
    print(f"Exported class mapping to {class_indices_path}")

    # 2. Build Model
    model = build_ricecare_model(
        input_shape=(224, 224, 3),
        num_classes=len(class_labels),
        freeze_base=True
    )

    # 3. Compile Model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Compiled RiceCare AI Model with Adam optimizer.")
    model.summary()

    # 4. Production Callbacks
    keras_model_path = os.path.join(models_dir, "ricecare_ai_model.keras")
    h5_model_path = os.path.join(models_dir, "ricecare_ai_model.h5")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=keras_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Fit Model
    print(f"Training RiceCare AI for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save in both .keras and .h5 formats
    print(f"Saving primary model to {keras_model_path}...")
    model.save(keras_model_path)

    print(f"Saving legacy model to {h5_model_path}...")
    try:
        model.save(h5_model_path)
    except Exception as e:
        print(f"H5 export notice: {e}")

    # Also keep alias for rice_leaf_disease_model for backward compatibility
    try:
        model.save(os.path.join(models_dir, "rice_leaf_disease_model.keras"))
        model.save(os.path.join(models_dir, "rice_leaf_disease_model.h5"))
    except Exception:
        pass

    # 7. Plot and save history
    plot_history_curves(history, save_path=os.path.join(assets_dir, "training_history.png"))
    print("RiceCare AI model training successfully completed.")
    return model, history

def plot_history_curves(history, save_path="assets/training_history.png"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#1b4332", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#2d6a4f", linewidth=2)
    ax1.set_title("RiceCare AI Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#c1121f", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#780000", linewidth=2)
    ax2.set_title("RiceCare AI Crossentropy Loss Across Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    train_ricecare_model()
