
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. LOAD DATA
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

data = np.load(
    PROJECT_DIR / "outputs" / "prepared_data.npz"
)

pred_data = np.load(
    PROJECT_DIR / "outputs" / "validation_predictions.npz"
)

X_val = data["X_val"]
y_true = pred_data["y_true"]
probabilities = pred_data["probabilities"]

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================
# 2. FIND FALSE POSITIVES AT 0.50
# =====================================

threshold = 0.50

predictions = (probabilities >= threshold).astype(int)

fp_indices = np.where(
    (y_true == 0) & (predictions == 1)
)[0]

print("=" * 55)
print(" FALSE POSITIVE INVESTIGATION")
print("=" * 55)

print("\nFalse positive indices:", fp_indices.tolist())

# =====================================
# 3. PROBABILITY DISTRIBUTION
# =====================================

positive_probs = probabilities[y_true == 1]
negative_probs = probabilities[y_true == 0]

print("\n--- PROBABILITY ANALYSIS ---")

print(
    "Minimum positive probability:",
    float(positive_probs.min())
)

print(
    "Maximum negative probability:",
    float(negative_probs.max())
)

print(
    "False positive probabilities:",
    probabilities[fp_indices].tolist()
)

print(
    "Positive sample count:",
    len(positive_probs)
)

print(
    "Negative sample count:",
    len(negative_probs)
)

# =====================================
# 4. INSPECT FALSE POSITIVE LIGHT CURVE
# =====================================

for index in fp_indices:

    flux = X_val[index, :, 0]

    probability = probabilities[index]

    print("\nFalse positive sample:", int(index))
    print("Actual label:", int(y_true[index]))
    print("Predicted probability:", float(probability))

    plt.figure(figsize=(12, 4))

    plt.plot(flux, linewidth=0.7)

    plt.title(
        f"False Positive - Validation Sample {index}"
    )

    plt.xlabel("Time / Observation Index")
    plt.ylabel("Normalized Flux")

    plt.grid(alpha=0.25)

    plt.tight_layout()

    save_path = (
        OUTPUT_DIR /
        f"false_positive_{index}.png"
    )

    plt.savefig(save_path, dpi=150)

    plt.close()

    print("Light curve saved:", save_path)

# =====================================
# 5. SAVE PROBABILITY DISTRIBUTION
# =====================================

plt.figure(figsize=(10, 5))

plt.hist(
    negative_probs,
    bins=40,
    alpha=0.65,
    label="Non-Planet"
)

plt.hist(
    positive_probs,
    bins=20,
    alpha=0.65,
    label="Exoplanet"
)

plt.axvline(
    0.50,
    linestyle="--",
    label="Threshold 0.50"
)

plt.axvline(
    0.53,
    linestyle=":",
    label="Candidate Threshold 0.53"
)

plt.xlabel("Predicted Exoplanet Probability")
plt.ylabel("Number of Samples")

plt.title("Validation Probability Distribution")

plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

distribution_path = (
    OUTPUT_DIR / "validation_probability_distribution.png"
)

plt.savefig(distribution_path, dpi=150)

plt.close()

print("\nDistribution saved:", distribution_path)

print("\nInvestigation completed.")
