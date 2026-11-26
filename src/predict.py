
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path


# ---------------- PATHS & CONFIG ----------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "exoplanet_model.pth"

DEVICE = torch.device("cpu")
THRESHOLD = 0.747613
EXPECTED_LENGTH = 3197


# ---------------- CNN ARCHITECTURE ----------------

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
        x = self.features(x)
        return self.classifier(x).squeeze(1)


# ---------------- LOAD TRAINED MODEL ----------------

@torch.no_grad()
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = ExoplanetCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ---------------- NORMALIZATION ----------------

def normalize_light_curve(flux):

    flux = np.asarray(flux, dtype=np.float32)

    if flux.ndim != 1:
        raise ValueError("Input must be a one-dimensional light curve.")

    if len(flux) != EXPECTED_LENGTH:
        raise ValueError(
            f"Expected {EXPECTED_LENGTH} measurements, got {len(flux)}."
        )

    if not np.all(np.isfinite(flux)):
        raise ValueError(
            "Input contains missing or infinite values."
        )

    median = np.median(flux)

    q75 = np.percentile(flux, 75)
    q25 = np.percentile(flux, 25)

    scale = q75 - q25

    if scale < 1e-6:
        scale = np.std(flux)

    scale = max(float(scale), 1e-6)

    normalized = (flux - median) / scale

    normalized = np.clip(normalized, -10, 10)

    return normalized.astype(np.float32)


# ---------------- PREDICTION ----------------

def predict_exoplanet(model, flux):

    normalized = normalize_light_curve(flux)

    # Shape: (batch, channels, sequence_length)
    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        logits = model(tensor)

        probability = torch.sigmoid(logits).item()

    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": THRESHOLD,
        "normalized_flux": normalized
    }
