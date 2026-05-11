import warnings
warnings.simplefilter("ignore")

import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Input
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# CONFIG
@dataclass
class Config:
    data_folder: str = "llis_20250315"
    time_windows: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    history_window_minutes: int = 10
    hk_square: dict = field(default_factory=lambda: {
        'lat_min': 22.15, 'lat_max': 22.55,
        'lon_min': 113.85, 'lon_max': 114.45
    })
    epochs: int = 50
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    l2_reg: float = 0.002
    threshold: float = 0.5
    lstm_units: list = field(default_factory=lambda: [64, 32])


config = Config()


# DATA LOADING
def load_data(folder_path: str):
    dataframes = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                df = pd.read_csv(file_path, sep=r'\s+', names=[
                    'version', 'year', 'month', 'day', 'hour', 'minutes', 'seconds', 'nanoseconds',
                    'latitude', 'longitude', 'peak_current', 'multiplicity', 'number_of_sensors', 'degrees_freedom',
                    'ellipse_angle', 'semi_major_axis', 'semi-minor_axis', 'chi_square', 'rise_time', 'peak_to_zero_time',
                    'max_rate_of_rise', 'cloud_indicator', 'angle_indicator', 'signal_indicator', 'timing_indicator'
                ])
                df = df[df['cloud_indicator'] == 0]  # Cloud-to-ground only
                df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minutes', 'seconds']])
                dataframes.append(df)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    data = pd.concat(dataframes, ignore_index=True)
    if data.empty:
        raise ValueError("No valid data loaded.")
    print(f"Loaded {len(data)} cloud-to-ground strikes.")
    return data


# FEATURE ENGINEERING
def create_sequences(data: pd.DataFrame, config: Config):
    """
    For every minute in the data range, extract:
        a sequence of feature vectors (one per minute) covering the history window
        a binary label indicating whether a strike occurs in the prediction window
    """
    time_step = timedelta(minutes=1)
    min_time = data['datetime'].min()
    max_time = data['datetime'].max()

    sequence_features = {window: [] for window in config.time_windows}
    sequence_labels = {window: [] for window in config.time_windows}

    current_time = min_time 

    while current_time <= max_time:
        # History window: from (current_time - history window) to current_time
        history_start = current_time - timedelta(minutes=config.history_window_minutes)

        # For each prediction window 
        for window in config.time_windows:
            pred_end = current_time + timedelta(minutes=window)

            # Build the sequence: one feature vector for each minute in the history window
            seq_time = history_start 
            minute_vectors = []
            while seq_time < current_time:
                # Extract strikes that occured in this specific minute
                minute_data = data[(data['datetime'] >= seq_time) & 
                                   (data['datetime'] < seq_time + time_step)]

                # Spatial filter (HK)
                strikes_in_square = minute_data[
                    (config.hk_square['lat_min'] <= minute_data['latitude']) &
                    (minute_data['latitude'] <= config.hk_square['lat_max']) &
                    (config.hk_square['lon_min'] <= minute_data['longitude']) &
                    (minute_data['longitude'] <= config.hk_square['lon_max'])
                ]

                feature_dict = {
                    'strike_count': len(strikes_in_square),
                    'avg_peak_current': strikes_in_square['peak_current'].abs().mean() if len(strikes_in_square) > 0 else 0,
                    'max_peak_current': strikes_in_square['peak_current'].abs().max() if len(strikes_in_square) > 0 else 0,
                    'std_peak_current': strikes_in_square['peak_current'].abs().std() if len(strikes_in_square) > 1 else 0,
                    'avg_num_sensors': strikes_in_square['number_of_sensors'].mean() if len(strikes_in_square) > 0 else 2,
                    'avg_chi_square': strikes_in_square['chi_square'].mean() if len(strikes_in_square) > 0 else 0,
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'is_weekend': 1 if current_time.weekday() >= 5 else 0,
                }
                minute_vectors.append(list(feature_dict.values()))
                seq_time += time_step
            
            # Sequence for this window : (history_indow_minutes, n_features)
            sequence_features[window].append(minute_vectors)
            
            # Label : 1 if any strike in prediction window inside HK box
            pred_strikes = data[(data['datetime'] >= current_time) & (data['datetime'] <pred_end)]
            pred_strikes_hk = pred_strikes [
                (config.hk_square['lat_min'] <= pred_strikes['latitude']) &
                (pred_strikes['latitude'] <= config.hk_square['lat_max']) &
                (config.hk_square['lon_min'] <= pred_strikes['longitude']) &
                (pred_strikes['longitude'] <= config.hk_square['lon_max'])
            ]
            label = 1 if len(pred_strikes_hk) > 0 else 0
            sequence_labels[window].append(label)

        current_time += time_step

    return sequence_features, sequence_labels


# MODEL BUILDING
def build_model(input_shape, config: Config):
    """
    Build a two-layer LSTM model
    input_shape: (time_steps, n_features)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config.lstm_units[0], return_sequences=True,
             kernel_regularizer=regularizers.l2(config.l2_reg),
             recurrent_regularizer=regularizers.l2(config.l2_reg)),
        BatchNormalization(),
        Dropout(config.dropout_rate),

        LSTM(config.lstm_units[1], return_sequences=False,
             kernel_regularizer=regularizers.l2(config.l2_reg),
             recurrent_regularizer=regularizers.l2(config.l2_reg)),
        BatchNormalization(),
        Dropout(config.dropout_rate),

        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# MAIN PIPELINE
def main():
    print("Starting Lightning Strike Forecaster for Hong Kong...\n")

    data = load_data(config.data_folder)
    features_sequences, labels_sequences = create_sequences(data, config)

    models = {}
    results = []

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    for window in config.time_windows:
        print(f"\n{'='*60}")
        print(f"Training model for {window}-minute prediction window")
        print(f"{'='*60}")

        X = np.array(features_sequences[window]) # shape: (n_samples, history_window_minutes, n_features)
        y = np.array(labels_sequences[window])

        print(f"Dataset shape: {X.shape} | Positive samples: {y.sum()}/{len(y)} ({y.mean():.2%})")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features  (flatten -> scales -> reshape)
        n_samples, n_steps, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)

        scaler = StandardScaler()
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_test_scaled_flat = scaler.transform(X_test_flat)

        X_train_scaled = X_train_scaled_flat.reshape(n_samples, n_steps, n_features)
        X_test_scaled = X_test_scaled_flat.reshape(X_test.shape[0], n_steps, n_features)

        # Save scaler
        scaler_path = model_dir / f"scaler_{window}min.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Build & train model
        model = build_model((X.shape[1], X.shape[2]), config)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience,
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config.reduce_lr_patience,
                            min_lr=1e-6, verbose=1),
            TensorBoard(log_dir=f".logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}_window{window}")
        ]

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        model.fit(
            X_train_scaled, y_train,
            epochs=config.epochs,
            validation_data=(X_test_scaled, y_test),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose="auto"
        )

        # Evaluation
        y_pred_prob = model.predict(X_test_scaled)
        y_pred = (y_pred_prob > config.threshold).astype(int).flatten()

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 0
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nResults for {window}-min window:")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # Save model
        model_path = model_dir / f"lightning_model_{window}min.keras"
        model.save(model_path)
        print(f"Model and scaler saved for {window}min window.")

        models[window] = (model, scaler)
        results.append({
            'Window (min)': window,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc
        })

    # Final Comparison Table
    print(f"\n{'='*80}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*80}")
    comparison_df = pd.DataFrame(results)
    print(comparison_df.round(4))


if __name__ == "__main__":
    main()