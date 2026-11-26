
import json
import numpy as np
import torch

from pathlib import Path
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "outputs" / "prepared_data.npz"
MODEL_PATH = ROOT / "models" / "exoplanet_model.pth"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Frozen validation-based threshold
THRESHOLD = 0.747613

# Same CNN architecture used during training
class ExoplanetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
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
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)


def main():

    print("=" * 55)
    print("       EXODISCOVER AI - FINAL TEST EVALUATION")
    print("=" * 55)

    # Load prepared data
    data = np.load(DATA_PATH)

    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)

    print("\nTest shape:", X_test.shape)
    print("Test positives:", int(y_test.sum()))
    print("Test negatives:", int(len(y_test) - y_test.sum()))

    # IMPORTANT:
    # X_test is already normalized.
    # Do NOT normalize it again.

    device = torch.device("cpu")

    # Load trained model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model = ExoplanetCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nExisting trained model loaded successfully.")

    # Predict probabilities
    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    # Apply frozen threshold
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 45)
    print(f"Frozen threshold: {THRESHOLD:.6f}")
    print("=" * 45)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Non-Planet", "Exoplanet"],
            zero_division=0
        )
    )

    # Save predictions
    np.savez(
        OUTPUT_DIR / "final_test_predictions.npz",
        y_true=y_test,
        probabilities=probabilities,
        predictions=predictions,
        threshold=THRESHOLD,
        confusion_matrix=cm
    )

    # Save metrics
    metrics = {
        "threshold": THRESHOLD,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_samples": int(len(y_test)),
        "positive_samples": int(y_test.sum())
    }

    with open(
        OUTPUT_DIR / "final_test_metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nResults saved successfully:")
    print("outputs/final_test_predictions.npz")
    print("outputs/final_test_metrics.json")


if __name__ == "__main__":
    main()
