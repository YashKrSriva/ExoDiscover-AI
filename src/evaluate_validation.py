
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. PROJECT PATHS AND DEVICE
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "outputs" / "prepared_data.npz"
MODEL_PATH = PROJECT_DIR / "models" / "exoplanet_model.pth"

OUTPUT_PATH = (
    PROJECT_DIR / "outputs" / "validation_predictions.npz"
)

METRICS_PATH = (
    PROJECT_DIR / "outputs" / "validation_metrics.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print(" EXODISCOVER AI - VALIDATION EVALUATION")
print("=" * 55)

print("Device:", DEVICE)


# ==========================================
# 2. LOAD VALIDATION DATA
# ==========================================

data = np.load(DATA_PATH)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.int64)

print("\nValidation shape:", X_val.shape)
print("Positive samples:", int(y_val.sum()))
print("Negative samples:", int(len(y_val) - y_val.sum()))

# Original prepared data shape:
# (samples, sequence_length, channels)

# PyTorch Conv1d expects:
# (samples, channels, sequence_length)

X_tensor = torch.tensor(
    X_val.transpose(0, 2, 1),
    dtype=torch.float32
).to(DEVICE)


# ==========================================
# 3. EXACT ORIGINAL CNN ARCHITECTURE
# ==========================================

class ExoplanetCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1, 32,
                kernel_size=7,
                stride=2,
                padding=3
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32, 64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x).squeeze(1)


# ==========================================
# 4. LOAD EXISTING TRAINED MODEL
# ==========================================

model = ExoplanetCNN().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "model_state_dict not found in checkpoint."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nExisting trained model loaded successfully.")


# ==========================================
# 5. GENERATE VALIDATION PREDICTIONS
# ==========================================

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.cpu().numpy()

# Keep the original threshold for initial comparison.
threshold = 0.5

predictions = (
    probabilities >= threshold
).astype(np.int64)


# ==========================================
# 6. CALCULATE VALIDATION METRICS
# ==========================================

cm = confusion_matrix(
    y_val,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0
)

pr_auc = average_precision_score(
    y_val,
    probabilities
)

accuracy = float(
    np.mean(y_val == predictions)
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 55)
print(" VALIDATION RESULTS - THRESHOLD 0.50")
print("=" * 55)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["Non-Planet", "Exoplanet"],
        zero_division=0
    )
)


# ==========================================
# 8. SAVE VALIDATION PREDICTIONS
# ==========================================

np.savez_compressed(
    OUTPUT_PATH,

    y_true=y_val,

    probabilities=probabilities,

    predictions=predictions,

    threshold=np.array(threshold),

    confusion_matrix=cm
)


# ==========================================
# 9. SAVE VALIDATION METRICS
# ==========================================

metrics = {

    "threshold": threshold,

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1),

    "pr_auc": float(pr_auc),

    "true_negatives": int(tn),

    "false_positives": int(fp),

    "false_negatives": int(fn),

    "true_positives": int(tp),

    "validation_samples": int(len(y_val)),

    "positive_samples": int(y_val.sum()),

    "negative_samples": int(len(y_val) - y_val.sum())
}

with open(METRICS_PATH, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\nValidation predictions saved:")
print(OUTPUT_PATH)

print("\nValidation metrics saved:")
print(METRICS_PATH)

print("\nEvaluation completed successfully!")
