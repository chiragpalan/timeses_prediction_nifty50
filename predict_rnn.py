import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

# File paths
csv_path = "data/nifty50_data.csv"  # Path to Nifty50 CSV data
prediction_path = "data/prediction.csv"  # Path to save predictions
model_path = "model/rnn_model"  # Folder containing the trained model

# Load the trained RNN model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model

# Preprocess data for prediction
def preprocess_data(data, lookback=60):
    """
    Prepares the input data for prediction based on lookback window.
    """
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled_features = (features - features.mean(axis=0)) / features.std(axis=0)
    X = [scaled_features[-lookback:]]  # Get the most recent 'lookback' rows
    return np.array(X)

# Generate future predictions
def predict_future(csv_path, model_path):
    """
    Predict Nifty 50 values for the next 15, 30, 45, and 60 minutes.
    """
    # Load the latest CSV data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}.")
    data = pd.read_csv(csv_path)

    # Ensure the data has enough rows for lookback window
    if len(data) < 60:
        raise ValueError("Insufficient data: At least 60 rows are required for prediction.")

    # Load the trained model
    model = load_model(model_path)

    # Preprocess the data
    X = preprocess_data(data)

    # Make predictions
    predictions = model.predict(X)
    predictions = predictions.flatten()  # Flatten the output

    # Generate predictions for 15, 30, 45, and 60 minutes
    prediction_15 = predictions[0]  # Prediction for the next 15 minutes
    prediction_30 = predictions[1]  # Prediction for the next 30 minutes
    prediction_45 = predictions[2]  # Prediction for the next 45 minutes
    prediction_60 = predictions[3]  # Prediction for the next 60 minutes

    return [prediction_15, prediction_30, prediction_45, prediction_60]

# Save predictions to CSV
def save_predictions(predictions, prediction_path):
    """
    Append predictions to a CSV file with a timestamp.
    """
    # Prepare data for saving
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prediction_data = {
        'Timestamp': [current_time],
        '15_min': [predictions[0]],
        '30_min': [predictions[1]],
        '45_min': [predictions[2]],
        '60_min': [predictions[3]],
    }
    df = pd.DataFrame(prediction_data)

    # Append to CSV
    if os.path.exists(prediction_path):
        existing_df = pd.read_csv(prediction_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    # Save updated CSV
    updated_df.to_csv(prediction_path, index=False)
    print("Predictions saved successfully to prediction.csv.")

# Main execution
if __name__ == "__main__":
    try:
        # Generate predictions
        predictions = predict_future(csv_path, model_path)

        # Save predictions to CSV
        save_predictions(predictions, prediction_path)

    except Exception as e:
        print(f"Error during prediction: {e}")