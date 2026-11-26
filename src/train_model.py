
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 0.0005
PATIENCE = 7

torch.manual_seed(42)
np.random.seed(42)

print("=" * 55)
print("       EXODISCOVER AI - CNN TRAINING")
print("=" * 55)

# -------------------------------
# 2. LOAD PREPARED DATA
# -------------------------------

data = np.load(OUTPUT_DIR / "prepared_data.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Convert NumPy arrays to PyTorch tensors
# Input format: (samples, channels, sequence_length)

def make_loader(X, y, shuffle=False):
    features = torch.tensor(
        X.transpose(0, 2, 1),
        dtype=torch.float32
    )

    labels = torch.tensor(
        y,
        dtype=torch.float32
    )

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

# -------------------------------
# 3. BUILD 1D CNN
# -------------------------------

class ExoplanetCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=7,
                      stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5,
                      padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3,
                      padding=1),
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


model = ExoplanetCNN().to(DEVICE)

print("\nModel Architecture:")
print(model)

# -------------------------------
# 4. HANDLE CLASS IMBALANCE
# -------------------------------

positive = int(y_train.sum())
negative = int(len(y_train) - positive)

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nPositive samples:", positive)
print("Negative samples:", negative)
print("Positive class weight:", pos_weight.item())

# -------------------------------
# 5. TRAINING FUNCTIONS
# -------------------------------

def run_epoch(loader, training=False):

    model.train() if training else model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.set_grad_enabled(training):

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

            total_loss += loss.item() * len(labels)

            probabilities = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, np.array(all_labels), np.array(all_probs)


# -------------------------------
# 6. MODEL TRAINING
# -------------------------------

best_val_loss = float("inf")
best_weights = None
wait = 0
history = []

print("\nStarting CNN Training...\n")

for epoch in range(1, EPOCHS + 1):

    train_loss, _, _ = run_epoch(
        train_loader, training=True
    )

    val_loss, val_labels, val_probs = run_epoch(
        val_loader, training=False
    )

    val_ap = average_precision_score(
        val_labels, val_probs
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_pr_auc": val_ap
    })

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val PR-AUC: {val_ap:.4f}"
    )

    # Early stopping based on validation loss
    if val_loss < best_val_loss - 1e-5:

        best_val_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())
        wait = 0

    else:
        wait += 1

    if wait >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# -------------------------------
# 7. SAVE BEST MODEL
# -------------------------------

if best_weights is None:
    raise RuntimeError("Model training did not produce valid weights.")

model.load_state_dict(best_weights)

model_path = MODEL_DIR / "exoplanet_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {
        "input_channels": 1,
        "sequence_length": 3197,
        "architecture": "ExoplanetCNN"
    }
}, model_path)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

print("\nBest model saved:", model_path)

# -------------------------------
# 8. FINAL TEST EVALUATION
# -------------------------------

test_loss, true_labels, test_probs = run_epoch(
    test_loader, training=False
)

threshold = 0.5

predictions = (test_probs >= threshold).astype(int)

metrics = {
    "test_loss": float(test_loss),
    "accuracy": float(
        accuracy_score(true_labels, predictions)
    ),
    "precision": float(
        precision_score(true_labels, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(true_labels, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(true_labels, predictions, zero_division=0)
    ),
    "pr_auc": float(
        average_precision_score(true_labels, test_probs)
    ),
    "threshold": threshold,
    "test_positive_samples": int(true_labels.sum())
}

print("\n" + "=" * 55)
print("             FINAL TEST RESULTS")
print("=" * 55)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predictions))

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predictions,
        target_names=["Non-Planet", "Planet"],
        zero_division=0
    )
)

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

np.savez_compressed(
    OUTPUT_DIR / "test_predictions.npz",
    actual=true_labels,
    probabilities=test_probs,
    predicted=predictions
)

print("\nAll training outputs saved successfully!")
print("EXODISCOVER AI TRAINING COMPLETED!")
