
import pandas as pd

from src.predict import load_model, predict_exoplanet

print("=" * 50)
print("EXODISCOVER AI - PREDICTION TEST")
print("=" * 50)

try:
    # Load actual trained model
    model = load_model()

    # Load original, unnormalized test data
    df = pd.read_csv("data/exoTest.csv")

    # Select sample 372 (zero-based index)
    sample = df.iloc[372]

    actual_label = int(sample.iloc[0])
    raw_flux = sample.iloc[1:].to_numpy()

    # Generate actual CNN prediction
    result = predict_exoplanet(model, raw_flux)

    print("\nModel loaded successfully!")
    print("Sample index:", 372)
    print("Actual class:", "Exoplanet" if actual_label == 2 else "Non-Planet")
    print("Predicted class:",
          "Exoplanet Candidate" if result["prediction"] == 1 else "Non-Planet")
    print("Candidate probability:", f'{result["probability"]:.6f}')
    print("Decision threshold:", result["threshold"])

    print("\nPrediction pipeline working!")

except Exception as e:
    print("\nERROR:", str(e))
