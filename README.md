# 🪐 ExoDiscover AI
### AI-Powered Exoplanet Candidate Detection Using Deep Learning

ExoDiscover AI is a machine learning project that uses a **1D Convolutional Neural Network (CNN)** to analyze stellar light curves and identify potential exoplanet candidates.

The system learns patterns from NASA Kepler-labelled light-curve data and predicts whether a given signal resembles an exoplanet transit signal.

> **Important:** ExoDiscover AI identifies potential exoplanet candidates, not scientifically confirmed planets. Further astronomical validation and follow-up observations are required.

---

## ✨ Features

- **AI-Based Detection:** Uses a trained 1D CNN to analyze stellar brightness measurements.
- **Light Curve Analysis:** Visualizes stellar flux variations over time.
- **Detection Score:** Displays the model's prediction score.
- **Animated Scanning:** Provides a visual scanning animation during detection.
- **Batch Prediction:** Supports analyzing multiple light curves.
- **Model Performance Dashboard:** Displays accuracy, precision, recall, F1-score and confusion matrix.
- **CSV Support:** Allows users to upload light-curve datasets.
- **Scientific Transparency:** Clearly distinguishes AI predictions from confirmed exoplanet discoveries.

---

## 🧠 How It Works

The project follows this workflow:

1. **Input:** Receives stellar light-curve measurements.
2. **Preprocessing:** Applies median and IQR-based normalization.
3. **Deep Learning:** Passes the normalized signal through a 1D CNN.
4. **Prediction:** Calculates the model's output score.
5. **Classification:** Classifies the signal as an exoplanet candidate or non-planet based on the selected threshold.
6. **Visualization:** Displays the light curve and prediction results.

### Model Architecture

The ExoDiscover AI model uses:

- Conv1D layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Adaptive Average Pooling
- Dropout regularization
- Fully Connected Layers
- Sigmoid-based prediction

**Framework:** PyTorch

---

## 📊 Model Performance

The trained model was evaluated on a separate test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 99.82% |
| Precision | 83.33% |
| Recall | 100% |
| F1 Score | 90.91% |
| PR-AUC | 1.00 |

### Test Dataset Results

| Measurement | Count |
|---|---:|
| Total samples | 570 |
| Actual exoplanets | 5 |
| Actual non-planets | 565 |
| True positives | 5 |
| True negatives | 564 |
| False positives | 1 |
| False negatives | 0 |

**Important scientific limitation:** The test dataset contains only five positive exoplanet examples. Therefore, these metrics should not be interpreted as proof of general performance on all astronomical objects.

The model is a binary classifier trained on exoplanet and non-planet labels. It does not independently identify black holes, pulsars, supernovae, or other astronomical phenomena.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- NASA Kepler-labelled light-curve dataset

---

## 📁 Project Structure

```text
ExoDiscover-AI/
│
├── data/
│   ├── exoTrain.csv
│   └── exoTest.csv
│
├── models/
│   └── exoplanet_model.pth
│
├── src/
│   ├── audit_data.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_validation.py
│   ├── analyze_thresholds.py
│   └── predict.py
│
├── outputs/
│   └── final_test_metrics.json
│
├── app.py
├── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ExoDiscover-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd ExoDiscover-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Prepare the Dataset

Place the required dataset files inside the `data/` directory:

```text
data/
├── exoTrain.csv
└── exoTest.csv
```

The dataset must contain 3197 flux measurements and a `LABEL` column.

Label mapping:

- `1` = Non-Planet
- `2` = Exoplanet

The raw CSV files are excluded from this repository. Obtain the original dataset separately and place the files in the expected directory.

### 7. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔬 Scientific Methodology

ExoDiscover AI uses light-curve classification to identify signals that resemble exoplanet transit patterns.

A transit occurs when a planet passes in front of its host star, causing a small reduction in observed stellar brightness.

However, brightness variations can also occur due to:

- Eclipsing binary stars
- Stellar activity
- Instrumental noise
- Other astronomical phenomena

The current implementation does not include a complete periodicity-based transit verification pipeline or independent confirmation system.

A high model score is not equivalent to the probability that an object is a confirmed planet.

---

## 🚀 Future Improvements

- Periodicity detection and transit verification.
- Eclipsing binary rejection.
- Multi-class astronomical signal classification.
- Improved false-positive analysis.
- Integration with additional astronomical datasets.
- Model explainability and signal interpretation.
- Integration with real-time astronomical observation data.

---

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/YOUR_USERNAME

---

## 📜 License

This project is intended for educational and research purposes.

A formal open-source license can be added to specify reuse and distribution permissions.
