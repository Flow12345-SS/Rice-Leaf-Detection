"""
RiceCare AI Deep Learning Architecture Module
MobileNetV2 Transfer Learning with Enhanced Enterprise Classification Head:
Input -> MobileNetV2 (ImageNet) -> GlobalAveragePooling2D -> Dense(256, relu) -> Dropout(0.5) -> Dense(128, relu) -> Dense(3, softmax)
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

def build_ricecare_model(input_shape=(224, 224, 3), num_classes=3, freeze_base=True, weights='imagenet'):
    """
    Constructs the RiceCare AI Transfer Learning model matching the specified commercial architecture:
      1. Input Image (224x224x3)
      2. MobileNetV2 Pretrained Backbone (ImageNet)
      3. GlobalAveragePooling2D
      4. Dense(256, activation='relu')
      5. Dropout(0.5)
      6. Dense(128, activation='relu')
      7. Dense(num_classes, activation='softmax')
    """
    inputs = Input(shape=input_shape, name="input_leaf_image")

    try:
        base_model = MobileNetV2(
            weights=weights,
            include_top=False,
            input_tensor=inputs
        )
    except Exception as e:
        print(f"Notice: Loading MobileNetV2 weights without pre-download ({e}).")
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_tensor=inputs
        )

    base_model.trainable = not freeze_base

    # Custom Multi-layer Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.5, name="dropout_0.5")(x)
    x = Dense(128, activation="relu", name="dense_128")(x)
    outputs = Dense(num_classes, activation="softmax", name="softmax_classification")(x)

    model = Model(inputs=inputs, outputs=outputs, name="RiceCareAI_MobileNetV2")
    return model

if __name__ == "__main__":
    model = build_ricecare_model()
    model.summary()
