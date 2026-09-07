"""
Root-level wrapper for training RiceCare AI.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from train import train_ricecare_model

if __name__ == "__main__":
    train_ricecare_model()
