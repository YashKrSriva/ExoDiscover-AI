
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "outputs" / "prepared_data.npz"
PREDICTION_PATH = ROOT / "outputs" / "final_test_predictions.npz"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

def main():

    print("=" * 55)
    print("   EXODISCOVER AI - FALSE POSITIVE INVESTIGATION")
    print("=" * 55)

    # Load prepared test data
    data = np.load(DATA_PATH)

    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)

    # Load saved test predictions
    results = np.load(PREDICTION_PATH)

    probabilities = results["probabilities"]
    predictions = results["predictions"]
    threshold = float(results["threshold"])

    # Identify false positives
    fp_indices = np.where(
        (y_test == 0) & (predictions == 1)
    )[0]

    print("\nTest samples:", len(y_test))
    print("Frozen threshold:", threshold)
    print("False Positives found:", len(fp_indices))

    if len(fp_indices) == 0:
        print("\nNo False Positives found.")
        return

    for index in fp_indices:

        probability = float(probabilities[index])
        signal = X_test[index, :, 0]

        print("\n" + "-" * 45)
        print("False Positive Sample Index:", int(index))
        print("Actual Label: Non-Planet (0)")
        print("Predicted Label: Exoplanet (1)")
        print(f"Predicted Score: {probability:.6f}")
        print(f"Frozen Threshold: {threshold:.6f}")

        print("\nSignal Statistics:")
        print(f"Minimum: {signal.min():.4f}")
        print(f"Maximum: {signal.max():.4f}")
        print(f"Mean:    {signal.mean():.4f}")
        print(f"Std:     {signal.std():.4f}")

        # Plot light curve
        plt.figure(figsize=(14, 5))

        plt.plot(
            np.arange(len(signal)),
            signal,
            linewidth=0.7
        )

        plt.title(
            f"False Positive - Test Sample {index}\n"
            f"Predicted Score: {probability:.4f}"
        )

        plt.xlabel("Observation Index")
        plt.ylabel("Normalized Flux")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"test_false_positive_{index}.png"
        )

        plt.savefig(output_path, dpi=200)
        plt.close()

        print("\nLight curve saved:", output_path)

    print("\nInvestigation plots generated successfully.")


if __name__ == "__main__":
    main()
