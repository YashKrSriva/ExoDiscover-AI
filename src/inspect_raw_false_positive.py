
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "exoTest.csv"
PREPARED_PATH = ROOT / "outputs" / "prepared_data.npz"

SAMPLE_INDEX = 372

def main():

    print("=" * 55)
    print("   ORIGINAL RAW DATA INVESTIGATION")
    print("=" * 55)

    # Load original test CSV
    df = pd.read_csv(RAW_PATH)

    # Get original row corresponding to test sample
    raw_row = df.iloc[SAMPLE_INDEX]

    raw_signal = raw_row.drop("LABEL").to_numpy(dtype=float)

    # Load prepared data
    data = np.load(PREPARED_PATH)
    normalized_signal = data["X_test"][SAMPLE_INDEX, :, 0]

    print("\nSample index:", SAMPLE_INDEX)
    print("Original label:", raw_row["LABEL"])

    print("\nRAW SIGNAL STATISTICS")
    print("Minimum:", np.min(raw_signal))
    print("Maximum:", np.max(raw_signal))
    print("Mean:", np.mean(raw_signal))
    print("Median:", np.median(raw_signal))
    print("Standard deviation:", np.std(raw_signal))

    print("\nRAW DATA QUALITY")
    print("NaN values:", np.isnan(raw_signal).sum())
    print("Infinite values:", np.isinf(raw_signal).sum())
    print("Zero values:", np.sum(raw_signal == 0))

    print("\nRAW CONSECUTIVE CHANGES")
    raw_diff = np.abs(np.diff(raw_signal))

    print("Maximum change:", raw_diff.max())
    print("Mean change:", raw_diff.mean())

    print("\nNORMALIZED SIGNAL COMPARISON")
    print("Raw length:", len(raw_signal))
    print("Normalized length:", len(normalized_signal))
    print("Normalized minimum:", normalized_signal.min())
    print("Normalized maximum:", normalized_signal.max())
    print("Clipped points at -10:",
          np.sum(normalized_signal <= -9.999))

    # Print first 20 observations for comparison
    print("\nFIRST 20 RAW OBSERVATIONS")
    print(np.round(raw_signal[:20], 4))

    print("\nFIRST 20 NORMALIZED OBSERVATIONS")
    print(np.round(normalized_signal[:20], 4))


if __name__ == "__main__":
    main()
