
import pandas as pd
import numpy as np

print("===== EXODISCOVER AI: DATA AUDIT =====")

train = pd.read_csv("data/exoTrain.csv")
test = pd.read_csv("data/exoTest.csv")

# 1. Class distribution
print("\nTRAINING CLASS DISTRIBUTION")
print(train["LABEL"].value_counts().sort_index())

print("\nTESTING CLASS DISTRIBUTION")
print(test["LABEL"].value_counts().sort_index())

# 2. Missing values
print("\nMISSING VALUES")
print("Train:", train.isna().sum().sum())
print("Test:", test.isna().sum().sum())

# 3. Infinite values
train_flux = train.iloc[:, 1:].to_numpy(dtype=np.float64)
test_flux = test.iloc[:, 1:].to_numpy(dtype=np.float64)

print("\nINFINITE VALUES")
print("Train:", np.isinf(train_flux).sum())
print("Test:", np.isinf(test_flux).sum())

# 4. Label verification
print("\nUNIQUE LABELS")
print("Train:", sorted(train["LABEL"].unique()))
print("Test:", sorted(test["LABEL"].unique()))

# 5. Dataset status
print("\nDATASET AUDIT COMPLETED!")
