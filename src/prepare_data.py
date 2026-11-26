
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("       EXODISCOVER AI - PREPROCESSING")
print("=" * 50)

# Load NASA Kepler dataset
train_df = pd.read_csv(DATA_DIR / "exoTrain.csv")
test_df = pd.read_csv(DATA_DIR / "exoTest.csv")

# Separate labels and flux measurements
X = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
y = (train_df["LABEL"].to_numpy() == 2).astype(np.int64)

X_test = test_df.iloc[:, 1:].to_numpy(dtype=np.float32)
y_test = (test_df["LABEL"].to_numpy() == 2).astype(np.int64)

print("\nOriginal Training Shape:", X.shape)
print("Original Testing Shape:", X_test.shape)

# Normalize each star's light curve independently
def normalize_light_curves(data):
    median = np.median(data, axis=1, keepdims=True)

    q75 = np.percentile(data, 75, axis=1, keepdims=True)
    q25 = np.percentile(data, 25, axis=1, keepdims=True)

    scale = q75 - q25

    # Fallback for constant or near-constant curves
    std = np.std(data, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, std, scale)
    scale = np.maximum(scale, 1e-6)

    normalized = (data - median) / scale

    # Numerical stability
    normalized = np.clip(normalized, -10, 10)

    return normalized.astype(np.float32)

X = normalize_light_curves(X)
X_test = normalize_light_curves(X_test)

# Add channel dimension for 1D CNN
X = X[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Stratified train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save prepared data
np.savez_compressed(
    OUTPUT_DIR / "prepared_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

print("\n===== PREPROCESSING RESULTS =====")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nPositive examples:")
print("Train:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))

print("\nNegative examples:")
print("Train:", int((y_train == 0).sum()))
print("Validation:", int((y_val == 0).sum()))
print("Test:", int((y_test == 0).sum()))

print("\nSaved: outputs/prepared_data.npz")
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
