# Lightning Strike Analysis Workflow

## Data Loading and Preprocessing
- Reads lightning strike data from files in the `llis_20250315` folder.
- Filters for cloud-to-ground strikes.
- Creates a datetime column for temporal analysis.

## Feature Engineering
- Processes data in 1-minute time steps for each time window (1, 2, 3, 4, 5 minutes).
- Extracts features from a 10-minute historical window.
- Creates binary labels indicating whether a strike occurs in the next window minutes within the HK square.

## Model Training
- Trains a separate Multi-Layer Perceptron (MLP) model for each time window.
- Uses features and labels specific to each window.
- Applies a train-test split tailored to each time window.

## Evaluation and Prediction
- Evaluates each model on its respective test set.
- Saves each trained model.
- Prints predictions and accuracy metrics, including a detailed analysis of the first 10 samples for each model.

## Output
- Displays the shape of features and labels for each time window.
- Reports test loss and accuracy metrics.
- Provides detailed prediction results for each time window.