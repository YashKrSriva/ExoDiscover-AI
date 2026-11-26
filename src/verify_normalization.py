
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "exoTest.csv"
PREPARED_PATH = ROOT / "outputs" / "prepared_data.npz"

SAMPLE_INDEX = 372

def main():

    df = pd.read_csv(RAW_PATH)

    raw = df.iloc[SAMPLE_INDEX].drop("LABEL").to_numpy(
        dtype=np.float64
    )

    data = np.load(PREPARED_PATH)
    normalized = data["X_test"][SAMPLE_INDEX, :, 0]

    # Calculate robust statistics
    median = np.median(raw)
    q1 = np.percentile(raw, 25)
    q3 = np.percentile(raw, 75)
    iqr = q3 - q1
    std = np.std(raw)

    print("=" * 55)
    print("NORMALIZATION VERIFICATION - SAMPLE 372")
    print("=" * 55)

    print("\nRAW STATISTICS")
    print("Median:", median)
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Standard deviation:", std)

    print("\nNORMALIZATION SCALE")
    if iqr > 1e-8:
        scale = iqr
        print("Scale method: IQR")
    else:
        scale = std
        print("Scale method: Standard deviation fallback")

    print("Scale:", scale)

    # Reconstruct normalization
    reconstructed = (raw - median) / scale
    reconstructed = np.clip(reconstructed, -10, 10)

    print("\nNORMALIZATION COMPARISON")
    print("Maximum absolute difference:",
          np.max(np.abs(reconstructed - normalized)))

    print("Reconstructed minimum:", reconstructed.min())
    print("Reconstructed maximum:", reconstructed.max())

    print("\nCLIPPING DETAILS")

    clipped_indices = np.where(reconstructed <= -10)[0]

    print("Lower clipped points:", len(clipped_indices))
    print("First 30 clipped indices:", clipped_indices[:30])

    # Longest consecutive clipped sequence
    if len(clipped_indices) > 0:

        groups = np.split(
            clipped_indices,
            np.where(np.diff(clipped_indices) != 1)[0] + 1
        )

        longest = max(groups, key=len)

        print("Longest consecutive clipped run:", len(longest))
        print("Run starts at:", int(longest[0]))
        print("Run ends at:", int(longest[-1]))

    print("\nRAW PERCENTILES")

    for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"{p:3d}th percentile:",
              np.percentile(raw, p))


if __name__ == "__main__":
    main()
