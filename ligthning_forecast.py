import warnings
warnings.simplefilter("ignore")

# 1. Import the Required Libraries and Define a Global Variable
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard , EarlyStopping
from keras.models import load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

folder_path = "llis_20250315"
time_windows = [1, 2, 3, 4, 5]  # Prediction windows in minutes
HK_SQUARE = {'lat_min': 22.15, 'lat_max': 22.55, 'lon_min': 113.85, 'lon_max': 114.45}

# 2. Load the data
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
        df = df[df['cloud_indicator'] == 0]  # Filter for cloud-to-ground lightning 
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minutes', 'seconds']])
        dataframes.append(df)
      except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

data = pd.concat(dataframes, ignore_index=True)
if data.empty:
  raise ValueError("No valid data loaded from the folder.")

# 3. Explore the data
print(data) # [3356 rows x 26 columns]

# 4. Feature engineering and labeling 
features_dict = {window: [] for window in time_windows}
labels_dict = {window: [] for window in time_windows}
time_step = timedelta(minutes=1)
min_time = data['datetime'].min()
max_time = data['datetime'].max()


for window in time_windows:
  current_time = min_time
  while current_time <= max_time:
      # Define a 10-minute historical window ending at current_time --- used to aggregate features from past lightning data
      window_start = current_time - timedelta(minutes=10)
      window_end = current_time
      # Define a 1/2../5-minute prediction window starting at current_time --- used to determine if a cloud-to-ground lightning event occurs
      pred_start = current_time
      pred_end = current_time + timedelta(minutes=window)  

      window_data = data[(data['datetime'] >= window_start) & (data['datetime'] < window_end)]
      pred_data = data[(data['datetime'] >= pred_start) & (data['datetime'] < pred_end)]

      # Spatial check: Filter data within the HK square
      strikes_in_square = window_data[
          (HK_SQUARE['lat_min'] <= window_data['latitude']) & (window_data['latitude'] <= HK_SQUARE['lat_max']) &
          (HK_SQUARE['lon_min'] <= window_data['longitude']) & (window_data['longitude'] <= HK_SQUARE['lon_max'])
      ]

      feature_dict = {
          'strike_count': len(strikes_in_square),
          'avg_peak_current': strikes_in_square['peak_current'].abs().mean() if len(strikes_in_square) > 0 else 0,
          'max_peak_current': strikes_in_square['peak_current'].abs().max() if len(strikes_in_square) > 0 else 0,
          'avg_num_sensors': strikes_in_square['number_of_sensors'].mean() if len(strikes_in_square) > 0 else 2,
          'avg_chi_square': strikes_in_square['chi_square'].mean() if len(strikes_in_square) > 0 else 0,
          'hour': current_time.hour,
          'day_of_week': current_time.weekday()
      }

      # Spatial check for prediction
      pred_strikes = pred_data[
          (HK_SQUARE['lat_min'] <= pred_data['latitude']) & (pred_data['latitude'] <= HK_SQUARE['lat_max']) &
          (HK_SQUARE['lon_min'] <= pred_data['longitude']) & (pred_data['longitude'] <= HK_SQUARE['lon_max'])
      ]
      label = 1 if len(pred_strikes) > 0 else 0

      features_dict[window].append([feature_dict[key] for key in feature_dict])
      labels_dict[window].append(label)

      current_time += time_step

X_dict = {window: np.array(features_dict[window]) for window in time_windows}
y_dict = {window: np.array(labels_dict[window]) for window in time_windows}

for window in time_windows:
    print(f"X shape for {window}-minute window: {X_dict[window].shape}") # (1389, 7)
    print(f"y shape for {window}-minute window: {y_dict[window].shape}") # (1389, )
    print(f"Label distribution for {window}-minute window: {np.bincount(y_dict[window])}") # [1346   43] == 1346 instances of label 0 and 43 instances of label 1.

# 5. Build Models
models = {} # Dictionary to store trained models and scalers
for window in time_windows:
    # Use the features and labels for the current time window
    X = X_dict[window]
    y = y_dict[window]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build the MLP model using Keras
    model = Sequential()
    model.add(Flatten(input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.002))) # L2 regularization
    model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(Dense(units=1, activation='sigmoid')) # Binary output

    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True)
    
    # 6. Compile the model
    adam_optimizer = Adam(learning_rate=0.001)  # Adaptive Moment Estimation optimizer
    model.compile(optimizer=adam_optimizer,
                  loss='binary_crossentropy',  # Binary cross-entropy for classification
                  metrics=['accuracy'])
    
    # 7. Train the model
    log_dir = ".logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") # Creates a directory path to store TensorBoard logs
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)  # Track metrics and visualize
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    training_history = model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_test_scaled, y_test),
                        callbacks=[tensorboard_callback],class_weight=class_weight_dict, verbose=1)
    

    # 8. Evaluate the model
    validation_loss, validation_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f'Test loss for {window}-minute window: {validation_loss:.4f}')
    print(f'Test accuracy for {window}-minute window: {validation_accuracy:.4f}')

    predictions = model.predict(X_test_scaled)
    prediction_results = (predictions > 0.5).astype(int).flatten()
    precision = precision_score(y_test, prediction_results)
    recall = recall_score(y_test, prediction_results)
    f1 = f1_score(y_test, prediction_results)
    roc_auc = roc_auc_score(y_test, predictions) if len(np.unique(y_test)) > 1 else 0
    
    print(f'Precision for {window}-minute window: {precision:.4f}')
    print(f'Recall for {window}-minute window: {recall:.4f}')
    print(f'F1-score for {window}-minute window: {f1:.4f}')
    print(f'ROC-AUC for {window}-minute window: {roc_auc:.4f}')
    
    # 9. Save the Model
    model_name = f'model/lightning_model_{window}min.h5'  
    if not os.path.exists('model'):  # Check if the folder exists
        os.makedirs('model')  
    model.save(model_name, save_format='h5')
    print(f"Model saved as {model_name}")

    # Store model and scaler
    loaded_model = load_model(model_name)
    models[window] = (loaded_model, scaler)

    # 10. Use the model 
    predictions = loaded_model.predict([X_test_scaled]) # Predict probabilities
    print('predictions:', predictions.shape) #  (278, 1)
    df1 = pd.DataFrame(predictions)
    print(df1)

    prediction_results =  (predictions > 0.5).astype(int)
    df2 = pd.DataFrame(prediction_results)
    print(df2)

    # Verify predictions against true labels
    correct_predictions = prediction_results.flatten() == y_test
    total_samples = len(y_test)
    correct = 0 
    correct_count = np.sum(correct_predictions)
    accuracy = correct_count / total_samples
    print(f'Accuracy for {window}-minute window: {accuracy:.4f} ({correct_count}/{total_samples})')

    print("\nDetailed Prediction Check (First 10 Samples):")
    for i in range(min(10, total_samples)):
        true_label = y_test[i]
        predicted_label = prediction_results[i][0]  # Access the single value if predictions is 2D
        is_correct = "Correct" if true_label == predicted_label else "Incorrect"
        if is_correct == "Correct":
            correct+=1
        else:
            correct+=0
        print(f"Sample {i}: True = {true_label}, Predicted = {predicted_label}, {is_correct}")






