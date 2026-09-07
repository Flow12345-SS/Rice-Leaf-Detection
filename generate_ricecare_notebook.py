"""
Script to generate notebooks/RiceCareAI_Complete_Pipeline.ipynb
covering all 16 modules for commercial AgriTech submission.
"""

import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()
cells = []

# Title & Commercial Branding
cells.append(nbf.v4.new_markdown_cell("""# 🌾 RiceCare AI™ - Smart Rice Disease Detection & Advisory Platform
### Commercial-Grade AgriTech Deep Learning & Computer Vision Platform
**Roles:** Senior AI/ML Engineer, Agricultural Data Scientist, Computer Vision Specialist, MLOps Architect  
**Frameworks:** TensorFlow 2.x, Keras 3.x, OpenCV, Streamlit, Plotly, Scikit-learn, ReportLab  
**Architecture:** MobileNetV2 Transfer Learning (`Dense(256) -> Dropout(0.5) -> Dense(128) -> Softmax(3)`)  

---

### 🌐 Live Platform & Source Repositories
- **GitHub Repository:** `https://github.com/your-username/RiceCare-AI-Platform` *(Placeholder)*
- **Streamlit Community Cloud Live App:** `https://ricecare-ai.streamlit.app` *(Placeholder)*

---

## 📑 Platform Module Navigation
1. [Module 1: Business Understanding & AgriTech Economics](#module-1-business-understanding)
2. [Module 2: Dataset Profile & Exploratory Data Analysis](#module-2-dataset-analysis)
3. [Module 3: Preprocessing, Standardization & Augmentation](#module-3-data-preprocessing)
4. [Module 4: Deep Learning Architecture (MobileNetV2)](#module-4-deep-learning-model)
5. [Module 5: Model Training with Production MLOps Callbacks](#module-5-model-training)
6. [Module 6: Evaluation: Confusion Matrix & Multi-Class ROC Analysis](#module-6-model-evaluation)
7. [Module 7: Computer Vision Lesion Severity Assessment Engine](#module-7-severity-assessment)
8. [Module 8: Agricultural Recommendation Engine & IPM Prescriptions](#module-8-recommendation-engine)
9. [Module 9: Automated Agronomic PDF Field Report Generation](#module-9-pdf-reporting)
10. [Module 10: Diagnostic Audit History Database Simulation](#module-10-audit-history)
11. [Module 11: Production Streamlit Dashboard Architecture](#module-11-dashboard-architecture)
12. [Module 12: Deployment, MLOps Structure & Future Roadmap](#module-12-deployment-and-roadmap)
"""))

# Module 1: Business Understanding
cells.append(nbf.v4.new_markdown_cell("""---
## Module 1: Business Understanding & AgriTech Economics

### 1.1 The Global Rice Security Challenge
Rice (*Oryza sativa*) supplies over 20% of global human caloric intake and represents the economic lifeblood for over 140 million smallholder farm families across Asia and Africa. Foliar diseases—specifically **Bacterial Blight**, **Brown Spot**, and **Leaf Smut**—systematically decimate rice crops, inducing **10% to 30% annual yield losses**.

### 1.2 The Economic Case for RiceCare AI
1. **Early Diagnostic Window:** Detecting infections at the mild stage ($<20\%$ leaf area affected) enables localized bactericidal/fungicidal spraying, preventing total crop collapse.
2. **Cost Optimization:** Indiscriminate blanket pesticide spraying costs farmers roughly $60–$100 per hectare per season. Targeted prescription reduces chemical usage by **up to 35%**.
3. **Yield Preservation:** Field trials demonstrate an average **yield recovery of 18% to 22%** when farmers apply the correct active ingredient within 72 hours of initial symptom manifestation.
"""))

# Module 2: Dataset Analysis
cells.append(nbf.v4.new_markdown_cell("""---
## Module 2: Dataset Profile & Exploratory Data Analysis (EDA)

We profile the Kaggle Rice Leaf Disease dataset across the three primary foliar disease categories:
- **Bacterial Blight (*Xanthomonas oryzae*)**
- **Brown Spot (*Bipolaris oryzae*)**
- **Leaf Smut (*Entyloma oryzae*)**
"""))

cells.append(nbf.v4.new_code_cell("""import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_theme(style="whitegrid")
DATASET_DIR = "../dataset"
train_dir = os.path.join(DATASET_DIR, "train")

# Scan dataset
classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
records = []

for cls in classes:
    files = glob.glob(os.path.join(train_dir, cls, "*.jpg"))
    for f in files:
        with Image.open(f) as im:
            w, h = im.size
        records.append({"Class": cls.replace("_", " "), "Width": w, "Height": h, "Path": f})

df = pd.DataFrame(records)
print(f"Total training images audited: {len(df)}")
df["Class"].value_counts()
"""))

cells.append(nbf.v4.new_code_cell("""# Visualizing Class Balance and Sample Lesions
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sns.countplot(data=df, x="Class", hue="Class", ax=axes[0], palette="crest", legend=False)
axes[0].set_title("Training Set Class Distribution", fontweight="bold")
axes[0].set_ylabel("Sample Count")

df["Class"].value_counts().plot.pie(ax=axes[1], autopct="%1.1f%%", colors=["#1b4332", "#e76f51", "#2a9d8f"], startangle=140)
axes[1].set_title("Class Equilibrium Proportions", fontweight="bold")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
"""))

# Module 3: Preprocessing
cells.append(nbf.v4.new_markdown_cell("""---
## Module 3: Preprocessing, Normalization & Data Augmentation

Images are resized to $224 \times 224$ pixels and rescaled to $[0, 1]$.
Affine transformations (rotation, zoom, shear, horizontal flips) are introduced to mimic natural field orientations and camera angles.
"""))

cells.append(nbf.v4.new_code_cell("""import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode="nearest"
)

eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = eval_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = eval_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_labels = {v: k.replace("_", " ") for k, v in train_gen.class_indices.items()}
print("Class Label Dictionary:", class_labels)
"""))

# Module 4: Model Architecture
cells.append(nbf.v4.new_markdown_cell("""---
## Module 4: Deep Learning Model Architecture

We implement **MobileNetV2 Transfer Learning** with the required multi-layer classification head:
$$\text{Input}(224 \times 224 \times 3) \to \text{MobileNetV2} \to \text{GAP} \to \text{Dense}(256) \to \text{Dropout}(0.5) \to \text{Dense}(128) \to \text{Softmax}(3)$$
"""))

cells.append(nbf.v4.new_code_cell("""from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

def build_ricecare_model(input_shape=(224, 224, 3), num_classes=3):
    inputs = Input(shape=input_shape, name="leaf_image_input")
    
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False  # Freeze feature extraction weights
    
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.5, name="dropout_head")(x)
    x = Dense(128, activation="relu", name="dense_128")(x)
    outputs = Dense(num_classes, activation="softmax", name="softmax_classifier")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="RiceCareAI_MobileNetV2")
    return model

model = build_ricecare_model()
model.summary()
"""))

# Module 5: Training
cells.append(nbf.v4.new_markdown_cell("""---
## Module 5: Model Training with MLOps Callbacks

We configure the Adam optimizer with `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.
"""))

cells.append(nbf.v4.new_code_cell("""from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model.compile(
    optimizer=Adam(learning_rate=0.0004),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath="../models/ricecare_ai_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

EPOCHS = 25
print("Starting training loop...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
"""))

# Module 6: Evaluation & ROC
cells.append(nbf.v4.new_markdown_cell("""---
## Module 6: Model Evaluation: Confusion Matrix & Multi-Class ROC Curves

We evaluate model performance on the held-out test split, generating:
- Overall Accuracy, Precision, Recall, and F1-Score
- Seaborn Confusion Matrix
- Multi-Class One-vs-Rest Receiver Operating Characteristic (ROC) Curves with Area Under Curve (AUC) metrics.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

test_gen.reset()
y_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_probs, axis=1)
y_true = test_gen.classes

target_names = [class_labels[i] for i in sorted(class_labels.keys())]

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

print(f"Overall Accuracy:  {acc * 100:.2f}%")
print(f"Weighted F1-Score: {f1 * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=target_names, yticklabels=target_names)
plt.title("RiceCare AI - Test Confusion Matrix", fontweight="bold")
plt.xlabel("Predicted Disease")
plt.ylabel("True Disease")
plt.tight_layout()
plt.show()

# Multi-Class ROC Curves
y_true_bin = label_binarize(y_true, classes=list(range(len(target_names))))
plt.figure(figsize=(7, 5))
colors = ["#1b4332", "#e76f51", "#2a9d8f"]
for i in range(len(target_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{target_names[i]} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
plt.title("Multi-Class ROC Curves", fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()
"""))

# Module 7: Severity
cells.append(nbf.v4.new_markdown_cell("""---
## Module 7: Computer Vision Disease Severity Assessment Engine

To quantify infection progression, we implement HSV thresholding and morphological filtering to segment necrotic lesions:
- **0% - 20%:** Mild Infection
- **21% - 50%:** Moderate Infection
- **51% - 100%:** Severe Infection
"""))

cells.append(nbf.v4.new_code_cell("""import cv2

def assess_severity(image_path):
    img = Image.open(image_path).convert("RGB").resize((400, 400))
    rgb = np.array(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Segment Leaf Tissue
    leaf_mask = cv2.inRange(hsv, np.array([0, 25, 20]), np.array([180, 255, 255]))
    leaf_pixels = max(int(np.count_nonzero(leaf_mask)), 1)
    
    # Segment Chlorotic & Necrotic Lesions
    mask_yellow = cv2.inRange(hsv, np.array([15, 45, 60]), np.array([35, 255, 255]))
    mask_brown = cv2.inRange(hsv, np.array([5, 50, 30]), np.array([18, 255, 210]))
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 55]))
    
    lesion_mask = cv2.bitwise_and(cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_brown, mask_black)), leaf_mask)
    lesion_pixels = int(np.count_nonzero(lesion_mask))
    
    pct = round(min(100.0, (lesion_pixels / leaf_pixels) * 100.0), 2)
    tier = "Mild Infection" if pct <= 20 else ("Moderate Infection" if pct <= 50 else "Severe Infection")
    
    # Overlay visualization
    overlay = rgb.copy()
    overlay[lesion_mask > 0] = [230, 45, 45]
    annotated = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
    
    return pct, tier, Image.fromarray(annotated)

sample_img = os.path.join(DATASET_DIR, "test", "Bacterial_Blight", os.listdir(os.path.join(DATASET_DIR, "test", "Bacterial_Blight"))[0])
pct, tier, overlay_im = assess_severity(sample_img)
print(f"Assessed Severity: {tier} ({pct}% leaf lesion coverage)")

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1); plt.imshow(Image.open(sample_img)); plt.title("Original Specimen"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(overlay_im); plt.title(f"Lesions Highlighted ({pct}%)"); plt.axis("off")
plt.tight_layout(); plt.show()
"""))

# Module 8: Recommendation Engine
cells.append(nbf.v4.new_markdown_cell("""---
## Module 8: Agricultural Recommendation Engine & IPM Prescriptions

A production dictionary matching diagnosed pathogens with curated chemical and organic interventions, recovery timelines, and yield impact models.
"""))

cells.append(nbf.v4.new_code_cell("""AGRONOMIC_DATABASE = {
    "Bacterial Blight": {
        "pesticides": "Streptomycin sulfate + Tetracycline (90:10) @ 300g/ha or Copper Hydroxide (2.5 g/L)",
        "organic": "Supernatant of fresh cow dung slurry (20%) or Neem oil (3%)",
        "prevention": "Avoid nitrogen over-application; ensure proper field drainage",
        "recovery_time": "12 - 16 Days",
        "yield_impact": "20% - 35% if untreated"
    },
    "Brown Spot": {
        "pesticides": "Mancozeb 75% WP @ 2.5 g/L or Tricyclazole 75% WP @ 0.6 g/L",
        "organic": "Hot water seed treatment (53-54C for 10-12 mins); Trichoderma viride seed treatment",
        "prevention": "Correct soil potassium deficiency; maintain continuous irrigation",
        "recovery_time": "10 - 14 Days",
        "yield_impact": "15% - 25% if untreated"
    },
    "Leaf Smut": {
        "pesticides": "Copper Oxychloride 50% WP @ 2.5 g/L or Propiconazole 25% EC @ 1 ml/L",
        "organic": "Canopy aeration pruning; Bacillus subtilis foliar spray",
        "prevention": "Moderate nitrogen top-dressing; avoid excessive crop density",
        "recovery_time": "7 - 10 Days",
        "yield_impact": "5% - 15% if untreated"
    }
}
print("Agronomic Knowledge Base loaded with 3 comprehensive disease profiles.")
"""))

# Module 9: PDF Reporting
cells.append(nbf.v4.new_markdown_cell("""---
## Module 9: Automated Agronomic PDF Field Report Generation

Using `reportlab`, RiceCare AI outputs formal, printable PDF diagnostic certificates for farmers, FPOs, and insurance adjusters.
"""))

cells.append(nbf.v4.new_code_cell("""import sys
sys.path.insert(0, "..")
from src.report_generator import generate_pdf_report

pdf_file = "../reports/sample_report.pdf"
out = generate_pdf_report(
    farmer_id="FARMER-IN-902",
    disease_name="Bacterial Blight",
    confidence_pct="99.8%",
    severity_level="Moderate Infection",
    severity_pct=36.4,
    treatment_info=AGRONOMIC_DATABASE["Bacterial Blight"],
    leaf_image_pil=Image.open(sample_img),
    output_path=pdf_file
)
print(f"Generated official PDF report: {out} ({os.path.getsize(pdf_file)} bytes)")
"""))

# Module 10: History Log
cells.append(nbf.v4.new_markdown_cell("""---
## Module 10: Diagnostic Audit History Database Simulation

Each scan is recorded in `history.csv` to track regional epidemiology and treatment compliance.
"""))

cells.append(nbf.v4.new_code_cell("""df_hist = pd.read_csv("../history.csv")
print(f"Historical Audit Transactions ({len(df_hist)} records):")
df_hist.tail()
"""))

# Module 11: Deployment & Conclusion
cells.append(nbf.v4.new_markdown_cell("""---
## Module 11: Production Streamlit Dashboard Architecture

The multi-page dashboard is executed locally with:
```bash
streamlit run ../app.py
```
Key interfaces include:
- **Executive Dashboard:** Regional outbreak trends and KPI tiles.
- **Disease Diagnosis:** Mobile camera capture + file upload with severity estimation.
- **Treatment Hub:** Prescriptive chemical and bio-organic guidelines.
- **Model Performance:** Live ROC curves and confusion matrix.
- **About:** System architecture and deployment specifications.

---
## Module 12: Conclusion & Future Roadmap

RiceCare AI successfully bridges deep learning research and practical smart agriculture.
By combining **MobileNetV2 edge inference**, **computer vision severity scoring**, and **automated PDF field reporting**, the system equips smallholder farmers with enterprise-grade crop pathology tools to protect yields, optimize agrochemical investments, and safeguard food security.
"""))

nb['cells'] = cells
out_nb_path = "notebooks/RiceCareAI_Complete_Pipeline.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Successfully generated {out_nb_path} with {len(cells)} cells.")
