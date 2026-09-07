"""
RiceCare AI Comprehensive Model Evaluation Module
Calculates Accuracy, Precision, Recall, F1 Score, Confusion Matrix,
and generates Multi-Class ROC Curves with AUC metrics.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath("src"))

from data_loader import get_data_generators

def evaluate_ricecare(model_path="models/ricecare_ai_model.keras", dataset_dir="dataset", assets_dir="assets"):
    os.makedirs(assets_dir, exist_ok=True)

    if not os.path.exists(model_path):
        alt = model_path.replace("ricecare_ai_model.keras", "rice_leaf_disease_model.keras")
        if os.path.exists(alt):
            model_path = alt

    # 1. Load Test Generator
    _, _, test_gen, class_labels = get_data_generators(dataset_dir=dataset_dir, batch_size=16)

    # 2. Load Model
    print(f"Loading model from {model_path} for evaluation...")
    model = tf.keras.models.load_model(model_path)

    # 3. Predict Probabilities
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    target_names = [class_labels[i] for i in sorted(class_labels.keys())]
    n_classes = len(target_names)

    # 4. Accuracy, Precision, Recall, F1
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print("\n" + "="*55)
    print("RICECARE AI EVALUATION ON UNSEEN TEST SAMPLES")
    print("="*55)
    print(f"Overall Accuracy:   {acc * 100:.2f}%")
    print(f"Weighted Precision: {prec * 100:.2f}%")
    print(f"Weighted Recall:    {rec * 100:.2f}%")
    print(f"Weighted F1-Score:  {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 5. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.title("RiceCare AI - Confusion Matrix (Test Split)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Disease Class", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    cm_path = os.path.join(assets_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # 6. Multi-Class ROC Curve and AUC
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # If 2 classes, handle shape
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = ["#1b4332", "#e76f51", "#2a9d8f", "#d90429"]

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i],
            tpr[i],
            color=colors[i % len(colors)],
            lw=2,
            label=f"{target_names[i]} (AUC = {roc_auc[i]:.3f})"
        )

    # Plot baseline diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Multi-Class Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    roc_path = os.path.join(assets_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")

    # 7. Save metrics JSON
    summary = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc_scores": {target_names[i]: float(roc_auc[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "classes": target_names
    }
    metrics_path = os.path.join(assets_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved metrics summary to {metrics_path}")

    return summary

if __name__ == "__main__":
    evaluate_ricecare()
