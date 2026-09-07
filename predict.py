"""
Root-level wrapper for RiceCare AI prediction.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_leaf_disease

if __name__ == "__main__":
    sample = "dataset/test/Bacterial_Blight/bacterial_blight_001.jpg"
    if os.path.exists(sample):
        res = predict_leaf_disease(sample)
        print("Root predict test:")
        print(json.dumps(res, indent=4))
    else:
        print("Test sample not found.")
