"""
Root level wrapper for severity assessment.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from severity import assess_disease_severity

if __name__ == "__main__":
    sample = "dataset/test/Bacterial_Blight/bacterial_blight_001.jpg"
    if os.path.exists(sample):
        res = assess_disease_severity(sample)
        print("Root severity assessment test:")
        print(f"Infected Area: {res['infected_percentage']}%")
        print(f"Severity Tier: {res['severity_level']}")
    else:
        print("Test sample not found.")
