"""
Prediction Pipeline Module
Provides reusable inference function `predict_leaf_disease(image_path)`
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# Global cached model and label map
_CACHED_MODEL = None
_CACHED_LABELS = None

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ricecare_ai_model.keras")
DEFAULT_LABEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "class_indices.json")

# Fallback classes if json is not present
FALLBACK_CLASSES = {0: "Bacterial_Blight", 1: "Brown_Spot", 2: "Leaf_Smut"}

def load_inference_artifacts(model_path=None, label_path=None):
    """
    Loads and caches model and class label mapping.
    """
    global _CACHED_MODEL, _CACHED_LABELS

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if label_path is None:
        label_path = DEFAULT_LABEL_PATH

    # Check fallback for .h5 if .keras not found
    if not os.path.exists(model_path):
        alt_path = model_path.replace(".keras", ".h5")
        if os.path.exists(alt_path):
            model_path = alt_path

    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")
        print(f"Loading model into memory: {model_path}")
        _CACHED_MODEL = tf.keras.models.load_model(model_path)

    if _CACHED_LABELS is None:
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                data = json.load(f)
                _CACHED_LABELS = {int(k): v for k, v in data.items()}
        else:
            _CACHED_LABELS = FALLBACK_CLASSES

    return _CACHED_MODEL, _CACHED_LABELS

def preprocess_image(image_input, target_size=(224, 224)):
    """
    Preprocesses input image path, PIL Image, or file buffer into model-ready tensor.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        # Assumed to be file buffer (e.g. Streamlit UploadedFile)
        img = Image.open(image_input).convert("RGB")

    # Resize to MobileNetV2 input dimensions
    img_resized = img.resize(target_size)
    # Convert to float numpy array
    img_array = np.array(img_resized, dtype=np.float32)
    # Normalization (0 - 1)
    img_array = img_array / 255.0
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch, img

def predict_leaf_disease(image_path, model_path=None):
    """
    Reusable prediction function.
    
    Args:
        image_path: Path to image file, PIL Image, or file stream.
        model_path: Optional custom path to model weights.
        
    Returns:
        dict: {
            'disease_name': str,
            'confidence_score': float, # between 0.0 and 1.0
            'confidence_percentage': str, # e.g. "98.42%"
            'class_probabilities': dict # {class_name: prob}
        }
    """
    model, label_map = load_inference_artifacts(model_path=model_path)
    processed_tensor, _ = preprocess_image(image_path)

    # Model inference
    predictions = model.predict(processed_tensor, verbose=0)[0]
    best_class_idx = int(np.argmax(predictions))
    best_confidence = float(predictions[best_class_idx])

    predicted_disease = label_map.get(best_class_idx, f"Class_{best_class_idx}")

    class_probs = {
        label_map.get(i, f"Class_{i}"): float(predictions[i])
        for i in range(len(predictions))
    }

    return {
        "disease_name": predicted_disease.replace("_", " "),
        "raw_label": predicted_disease,
        "confidence_score": best_confidence,
        "confidence_percentage": f"{best_confidence * 100:.2f}%",
        "class_probabilities": class_probs
    }

if __name__ == "__main__":
    # Test with a sample from test set
    sample_test = "dataset/test/Bacterial_Blight/bacterial_blight_001.jpg"
    if os.path.exists(sample_test):
        res = predict_leaf_disease(sample_test)
        print("Test Prediction:")
        print(json.dumps(res, indent=4))
    else:
        print("Sample test image not found.")
