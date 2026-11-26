
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================
# 1. PATHS
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR / "outputs" / "validation_predictions.npz"
)

OUTPUT_PATH = (
    PROJECT_DIR / "outputs" / "threshold_analysis.csv"
)

# =====================================
# 2. LOAD VALIDATION PREDICTIONS
# =====================================

data = np.load(INPUT_PATH)

y_true = data["y_true"].astype(int)
probabilities = data["probabilities"].astype(float)

print("=" * 55)
print(" EXODISCOVER AI - THRESHOLD ANALYSIS")
print("=" * 55)

print("\nValidation samples:", len(y_true))
print("Positive samples:", int(y_true.sum()))
print("Negative samples:", int(len(y_true) - y_true.sum()))

# =====================================
# 3. GENERATE THRESHOLD CANDIDATES
# =====================================

# Evaluate thresholds from 0.01 to 0.99
thresholds = np.unique(
    np.concatenate([
        np.arange(0.01, 1.00, 0.01),
        np.array([0.5]),
        probabilities
    ])
)

results = []

# =====================================
# 4. EVALUATE EACH THRESHOLD
# =====================================

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    results.append({

        "threshold": float(threshold),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "true_negatives": int(tn),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "true_positives": int(tp)
    })

# =====================================
# 5. CREATE RESULTS TABLE
# =====================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# =====================================
# 6. SHOW BASELINE THRESHOLD
# =====================================

baseline = df[
    np.isclose(df["threshold"], 0.5)
].iloc[0]

print("\n--- CURRENT THRESHOLD: 0.50 ---")

print(baseline.to_string())

# =====================================
# 7. BEST F1 THRESHOLD
# =====================================

best_f1 = df.sort_values(
    ["f1_score", "precision", "threshold"],
    ascending=[False, False, False]
).iloc[0]

print("\n--- BEST F1 CANDIDATE ---")

print(best_f1.to_string())

# =====================================
# 8. HIGHEST THRESHOLD WITH 100% RECALL
# =====================================

full_recall = df[
    np.isclose(df["recall"], 1.0)
]

if len(full_recall) > 0:

    best_recall_threshold = full_recall.sort_values(
        ["precision", "threshold"],
        ascending=[False, False]
    ).iloc[0]

    print("\n--- BEST PRECISION WITH 100% RECALL ---")

    print(best_recall_threshold.to_string())

else:

    print("\nNo threshold achieved 100% recall.")

# =====================================
# 9. TOP 10 THRESHOLD CANDIDATES
# =====================================

print("\n--- TOP 10 CANDIDATES BY F1 ---")

top_results = df.sort_values(
    ["f1_score", "precision"],
    ascending=[False, False]
).head(10)

print(
    top_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# =====================================
# 10. FALSE POSITIVE SAMPLE AT 0.50
# =====================================

baseline_predictions = (
    probabilities >= 0.5
).astype(int)

false_positive_indices = np.where(
    (y_true == 0) &
    (baseline_predictions == 1)
)[0]

print("\n--- FALSE POSITIVE SAMPLE INDICES ---")

print(false_positive_indices.tolist())

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nThreshold analysis completed.")
print(
    "Note: These are validation candidates, "
    "not final test results."
)
