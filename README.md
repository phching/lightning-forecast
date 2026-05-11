# Lightning Strike Forecaster — Hong Kong

A machine learning pipeline that predicts **cloud-to-ground lightning strikes** over Hong Kong up to 5 minutes in advance, using historical LLIS sensor data and an LSTM neural network.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

This project trains a separate binary classifier for each of five prediction windows (1, 2, 3, 4, and 5 minutes). Given a 10-minute history of lightning activity(one feature vector per minutes) within a bounding box covering Hong Kong, the LSTM model predicts whether a cloud-to-ground strike will occur in the next N minutes.
The use of LSTM allowss the model to capture temporal patterns in lightning activity over the history window.

| Prediction Window | Task |
|---|---|
| 1 minute | Very short-term nowcasting |
| 2 – 3 minutes | Short-term warning |
| 4 – 5 minutes | Early advisory |

---

## Features

- Loads and parses raw LLIS (Lightning Location and Information System) data files
- Filters for cloud-to-ground strikes only (`cloud_indicator = 0`)
- Creates temporal seuences with 1-minute resolution over a 10-minute history window
- Spatial filtering to the Hong Kong bounding box
- Handles severe class imbalance via computed class weights
- LSTM architecture with BatchNormalization, Dropout, and L2 regularisation
- Early stopping and learning rate scheduling via callbacks
- Saves both the trained model (`.keras`) and fitted scaler (`.pkl`) per window
- Prints a final side-by-side comparison table across all five windows

---

## Project Structure

```
.
├── llis_20250315/               # Raw LLIS lightning data files (not included)
├── model/                       # Saved models and scalers (auto-created)
│   ├── lightning_model_1min.keras
│   ├── scaler_1min.pkl
│   └── ...
├── .logs/fit/                   # TensorBoard logs (auto-created)
├── lightning_forecast.py
└── README.md
```

---

## Requirements

- Python 3.10+
- TensorFlow / Keras 3.x
- scikit-learn
- pandas
- numpy

---

## Installation

```bash
# Clone the repository
git clone https://github.com/phching/lightning-forecast.git
cd lightning-forecast

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow scikit-learn pandas numpy
```

---

## Usage

1. Place your LLIS data files inside the `llis_20250315/` folder.
2. Run the pipeline:

```bash
uv run lightning_forecast.py
```

3. Monitor training in TensorBoard (optional):

```bash
tensorboard --logdir .logs/fit
```

Trained models and scalers are saved automatically to the `model/` directory.

### Loading a saved model for inference

```python
import pickle
import numpy as np
import keras

model  = keras.models.load_model("model/lightning_model_3min.keras")
with open("model/scaler_3min.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example : 10 minutes of history x 9 features
# features: [strike_count, avg_peak_current, max_peak_current, std_peak_current,
#            avg_num_sensors, avg_chi_square, hour, day_of_week, is_weekend]
history_sequence = np.random.rand(10, 9).astype(np.float32) # replace with real data
history_scaled = scaler.transform(history_sequence.reshape(-1,9)).reshape(1,10,9)

prob = model.predict(history_scaled, verbose=0)[0][0]
print(f"Probability of strike in next 3 minutes: {prob:.2%}")
```

---

## How It Works

```
Raw LLIS files
      │
      ▼
 Load & filter           cloud_indicator = 0 (CG strikes only)
      │
      ▼
 Sequence extraction      10-min history (1 vector per minute)
      │
      ▼
 Labelling               Did any CG strike occur in the next N minutes?
      │
      ▼
 Train / test split      Stratified 80/20 split per window
      │
      ▼
 StandardScaler          Fitted on train set only — applied to test set
      │
      ▼
LSTM training            Class weights · EarlyStopping · ReduceLROnPlateau
      │
      ▼
 Evaluation              Precision · Recall · F1 · ROC-AUC · confusion matrix
      │
      ▼
 Save                    .keras model + .pkl scaler (one pair per window)
```

### HK Bounding Box

| Boundary | Value |
|---|---|
| Latitude min | 22.15° N |
| Latitude max | 22.55° N |
| Longitude min | 113.85° E |
| Longitude max | 114.45° E |

### Input Features (9 total)

| Feature | Description |
|---|---|
| `strike_count` | Number of CG strikes in the past 10 minutes |
| `avg_peak_current` | Mean absolute peak current (kA) |
| `max_peak_current` | Maximum absolute peak current (kA) |
| `std_peak_current` | Standard deviation of peak current |
| `avg_num_sensors` | Average number of sensors that detected each strike |
| `avg_chi_square` | Average chi-square value (location accuracy indicator) |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0 = Monday, 6 = Sunday) |
| `is_weekend` | Binary flag: 1 if Saturday or Sunday |

---

## Model Architecture

```
Input (shape(10, 9)) # time steps x features
    │
    LSTM(64, return_sequences=True) + BatchNormalization + Dropout(0.3)
    │
    LSTM(32, return_sequences=False)  + BatchNormalization  + Dropout(0.3)
    │
    Dense(16)  + ReLU + Dropout(0.2)
    │
    Dense(1)   + Sigmoid
    │
Output: probability of CG strike in next N minutes
```

- Optimiser: Adam (lr = 0.001)
- Loss: Binary cross-entropy
- Regularisation: L2 (λ = 0.002) + Dropout
- Early stopping: patience = 7 epochs on val_loss
- LR scheduler: ReduceLROnPlateau (factor = 0.5, patience = 3)

---

## Evaluation Metrics

Because lightning events are rare (~3% of samples are positive), **accuracy alone is misleading**. The pipeline reports:

| Metric | Why it matters |
|---|---|
| Precision | Of all predicted strikes, how many were real? |
| Recall | Of all real strikes, how many did we catch? |
| F1 score | Harmonic mean of precision and recall |
| ROC-AUC | Model's ability to rank positives above negatives |
| Confusion matrix | Breakdown of true/false positives and negatives |

Recall is the most safety-critical metric — missing a real strike is worse than a false alarm.

---

## Configuration

All hyperparameters are centralised in the `Config` dataclass at the top of the script:

```python
@dataclass
class Config:
    data_folder: str = "llis_20250315"
    time_windows: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    history_window_minutes: int = 10
    epochs: int = 50
    early_stopping_patience: int = 7
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    threshold: float = 0.5
    ...
```

No magic numbers are scattered across the code — edit one place to change any setting.

---

## Known Limitations

- **Small dataset** — the included data covers a single date (`20250315`), which limits generalisation and makes the class imbalance more pronounced.
- **No spatial resolution** — the model treats all strikes within the HK bounding box as equivalent; it does not predict *where* a strike will land.
- **Static threshold** — the 0.5 decision threshold is fixed; adjusting it can trade precision for recall depending on the use case.
- **No temporal cross-validation** — the train/test split is random, which can allow future data to leak into training. A time-based split would be more rigorous.

---

## License

This project is released under the [MIT License](LICENSE).