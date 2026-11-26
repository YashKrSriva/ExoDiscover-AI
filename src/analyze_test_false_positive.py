
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "outputs" / "prepared_data.npz"
PREDICTION_PATH = ROOT / "outputs" / "final_test_predictions.npz"

def main():

    data = np.load(DATA_PATH)
    results = np.load(PREDICTION_PATH)

    X_test = data["X_test"]
    y_test = data["y_test"].astype(int)
    probabilities = results["probabilities"]
    predictions = results["predictions"]

    fp_indices = np.where(
        (y_test == 0) & (predictions == 1)
    )[0]

    for index in fp_indices:

        signal = X_test[index, :, 0]

        print("\n" + "=" * 50)
        print("TEST FALSE POSITIVE ANALYSIS")
        print("=" * 50)

        print("Sample index:", index)
        print("Predicted score:", probabilities[index])

        # Clipping analysis
        lower_clipped = np.sum(signal <= -9.999)
        upper_clipped = np.sum(signal >= 9.999)

        print("\nCLIPPING ANALYSIS")
        print("Lower clipped points:", lower_clipped)
        print("Upper clipped points:", upper_clipped)

        # Signal statistics
        print("\nSIGNAL STATISTICS")
        print("Median:", np.median(signal))
        print("25th percentile:", np.percentile(signal, 25))
        print("75th percentile:", np.percentile(signal, 75))
        print("90th percentile:", np.percentile(signal, 90))
        print("10th percentile:", np.percentile(signal, 10))

        # Large jumps between consecutive observations
        differences = np.abs(np.diff(signal))

        print("\nCONSECUTIVE CHANGE ANALYSIS")
        print("Maximum change:", differences.max())
        print("Mean change:", differences.mean())
        print("Changes above 3:", np.sum(differences > 3))
        print("Changes above 5:", np.sum(differences > 5))

        # Split signal into 10 regions
        print("\n10-REGION SIGNAL ANALYSIS")

        regions = np.array_split(signal, 10)

        for i, region in enumerate(regions, start=1):
            print(
                f"Region {i}: "
                f"mean={region.mean():.3f}, "
                f"std={region.std():.3f}, "
                f"min={region.min():.3f}, "
                f"max={region.max():.3f}"
            )

if __name__ == "__main__":
    main()
