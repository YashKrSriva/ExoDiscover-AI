
import pandas as pd
import os

train_path = "data/exoTrain.csv"
test_path = "data/exoTest.csv"

if not os.path.exists(train_path):
    print("ERROR: exoTrain.csv not found!")
elif not os.path.exists(test_path):
    print("ERROR: exoTest.csv not found!")
else:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("\n===== EXODISCOVER AI =====")
    print("Training Shape:", train.shape)
    print("Testing Shape:", test.shape)

    print("\nClass Distribution:")
    print(train.iloc[:, 0].value_counts())

    print("\nFirst 5 Rows:")
    print(train.head())

    print("\nDataset inspection completed!")
