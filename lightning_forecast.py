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
from keras.layers import Dense, Dropout, BatchNormalization
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
def create_features_and_labels(data: pd.DataFrame, config: Config):
    features_dict = {window: [] for window in config.time_windows}
    labels_dict = {window: [] for window in config.time_windows}

    time_step = timedelta(minutes=1)
    min_time = data['datetime'].min()
    max_time = data['datetime'].max()

    for window in config.time_windows:
        current_time = min_time
        while current_time <= max_time:
            window_start = current_time - timedelta(minutes=config.history_window_minutes)
            pred_end = current_time + timedelta(minutes=window)

            # Historical window
            hist_data = data[(data['datetime'] >= window_start) & (data['datetime'] < current_time)]
            # Prediction window
            pred_data = data[(data['datetime'] >= current_time) & (data['datetime'] < pred_end)]

            # Spatial filter (HK)
            strikes_in_square = hist_data[
                (config.hk_square['lat_min'] <= hist_data['latitude']) &
                (hist_data['latitude'] <= config.hk_square['lat_max']) &
                (config.hk_square['lon_min'] <= hist_data['longitude']) &
                (hist_data['longitude'] <= config.hk_square['lon_max'])
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

            # Label
            pred_strikes = pred_data[
                (config.hk_square['lat_min'] <= pred_data['latitude']) &
                (pred_data['latitude'] <= config.hk_square['lat_max']) &
                (config.hk_square['lon_min'] <= pred_data['longitude']) &
                (pred_data['longitude'] <= config.hk_square['lon_max'])
            ]
            label = 1 if len(pred_strikes) > 0 else 0

            features_dict[window].append(list(feature_dict.values()))
            labels_dict[window].append(label)

            current_time += time_step

    return features_dict, labels_dict


# MODEL BUILDING
def build_model(input_dim: int, config: Config):
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(config.l2_reg), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(config.dropout_rate),

        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(config.l2_reg)),
        BatchNormalization(),
        Dropout(config.dropout_rate),

        Dense(32, activation='relu'),
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
    features_dict, labels_dict = create_features_and_labels(data, config)

    models = {}
    results = []

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    for window in config.time_windows:
        print(f"\n{'='*60}")
        print(f"Training model for {window}-minute prediction window")
        print(f"{'='*60}")

        X = np.array(features_dict[window])
        y = np.array(labels_dict[window])

        print(f"Dataset shape: {X.shape} | Positive samples: {y.sum()}/{len(y)} ({y.mean():.2%})")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        scaler_path = model_dir / f"scaler_{window}min.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Build & train model
        model = build_model(X_train.shape[1], config)

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
            verbose=1
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