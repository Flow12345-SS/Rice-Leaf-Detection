"""
Computer Vision Disease Severity Assessment Module
Quantifies foliar lesion area percentage using HSV color thresholding and morphological operations.

Severity Levels:
- 0% - 20%: Mild Infection
- 21% - 50%: Moderate Infection
- 51% - 100%: Severe Infection
"""

import cv2
import numpy as np
from PIL import Image

def assess_disease_severity(image_input):
    """
    Analyzes an input rice leaf image, computes infected area percentage,
    determines severity tier, and produces an annotated diagnostic overlay.
    
    Args:
        image_input: File path (str), PIL Image, or numpy RGB array.
        
    Returns:
        dict: {
            'infected_percentage': float,
            'severity_level': str ('Mild Infection', 'Moderate Infection', 'Severe Infection'),
            'severity_badge_color': str (hex color),
            'leaf_pixels': int,
            'lesion_pixels': int,
            'overlay_image': PIL.Image, # Leaf with highlighted lesions
            'mask_image': PIL.Image    # Binary lesion mask
        }
    """
    # 1. Load image as RGB numpy array
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")

    # Resize to standard analysis resolution to maintain consistent morphological kernel scale
    img_resized = img.resize((400, 400))
    rgb_arr = np.array(img_resized)
    hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)

    # 2. Extract Whole Leaf Mask (Exclude light neutral background)
    # Background in dataset is generally light neutral/white (S < 30, V > 200)
    # Leaf tissue contains green, yellow, brown, or dark pixels
    lower_leaf = np.array([0, 25, 20])
    upper_leaf = np.array([180, 255, 255])
    leaf_mask = cv2.inRange(hsv_arr, lower_leaf, upper_leaf)

    # Clean leaf mask with morphological closing
    kernel_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_leaf)
    leaf_pixels = int(np.count_nonzero(leaf_mask))

    # Guard against invalid/empty images
    if leaf_pixels < 500:
        leaf_pixels = 400 * 400
        leaf_mask = np.ones((400, 400), dtype=np.uint8) * 255

    # 3. Detect Lesion Types:
    # A. Bacterial Blight / Chlorosis (Wavy yellow-tan marginal lesions)
    lower_yellow = np.array([15, 45, 60])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv_arr, lower_yellow, upper_yellow)

    # B. Brown Spot / Necrosis (Oval reddish-brown lesions)
    lower_brown = np.array([5, 50, 30])
    upper_brown = np.array([18, 255, 210])
    mask_brown = cv2.inRange(hsv_arr, lower_brown, upper_brown)

    # C. Leaf Smut (Dark charcoal / black pustules)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 55])
    mask_black = cv2.inRange(hsv_arr, lower_black, upper_black)

    # Combine necrotic disease indicators constrained to leaf blade
    combined_lesion = cv2.bitwise_or(mask_yellow, mask_brown)
    combined_lesion = cv2.bitwise_or(combined_lesion, mask_black)
    lesion_mask = cv2.bitwise_and(combined_lesion, leaf_mask)

    # Morphological noise removal
    kernel_lesion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel_lesion)

    lesion_pixels = int(np.count_nonzero(lesion_mask))

    # 4. Compute Percentage
    infected_pct = min(100.0, max(0.0, (lesion_pixels / leaf_pixels) * 100.0))

    # 5. Classify Severity
    if infected_pct <= 20.0:
        severity_level = "Mild Infection"
        badge_color = "#2e7d32"  # Green
        severity_class = "mild"
    elif infected_pct <= 50.0:
        severity_level = "Moderate Infection"
        badge_color = "#f57c00"  # Orange
        severity_class = "moderate"
    else:
        severity_level = "Severe Infection"
        badge_color = "#d32f2f"  # Red
        severity_class = "severe"

    # 6. Create Visual Diagnostic Overlay (Red tint over lesions)
    overlay = rgb_arr.copy()
    # Apply red mask to lesion pixels
    overlay[lesion_mask > 0] = [230, 45, 45]
    # Blend with original image for semi-transparent diagnostic view
    annotated = cv2.addWeighted(rgb_arr, 0.65, overlay, 0.35, 0)

    return {
        "infected_percentage": round(infected_pct, 2),
        "severity_level": severity_level,
        "severity_class": severity_class,
        "severity_badge_color": badge_color,
        "leaf_pixels": leaf_pixels,
        "lesion_pixels": lesion_pixels,
        "overlay_image": Image.fromarray(annotated),
        "mask_image": Image.fromarray(lesion_mask)
    }

if __name__ == "__main__":
    import os
    sample = "dataset/test/Bacterial_Blight/bacterial_blight_001.jpg"
    if os.path.exists(sample):
        res = assess_disease_severity(sample)
        print(f"Sample Analysis: {res['severity_level']} ({res['infected_percentage']}%)")
    else:
        print("Sample image not found for test.")
