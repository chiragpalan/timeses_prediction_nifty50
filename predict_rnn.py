import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

# File paths
csv_path = "data/nifty50_data.csv"  # Path to Nifty50 CSV data
prediction_path = "data/prediction.csv"  # Path to save predictions
model_path = "model/rnn_model"  # Folder containing the trained model

# Ensure the data folder exists
os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

# Create an empty prediction.csv if it doesn't exist
if not os.path.exists(prediction_path):
    pd.DataFrame(columns=['Timestamp', '15_min', '30_min', '45_min', '60_min']).to_csv(prediction_path, index=False)

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
    # Convert columns to numeric, coercing errors to NaN
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Handle NaN values by dropping rows with NaN in critical columns
    data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    # Ensure there are at least 'lookback' rows
    if len(data) < lookback:
        raise ValueError(f"Insufficient data: At least {lookback} rows are required for prediction.")

    # Normalize features
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled_features = (features - features.mean(axis=0)) / features.std(axis=0)

    # Get the most recent 'lookback' rows
    X = [scaled_features[-lookback:]]  # Get the most recent 'lookback' rows
    return np.array(X)

# Generate future predictions
# Generate future predictions
def predict_future(csv_path, model_path):
    """
    Predict Nifty 50 values for the next 15, 30, 45, and 60 minutes.
    """
    # Load the latest CSV data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}.")
    
    data = pd.read_csv(csv_path)

    # Drop the row containing ^NSE (assuming it's right below the header)
    if '^NSE' in data.iloc[0].values:
        data = data.drop(index=0).reset_index(drop=True)

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

    # Append to or create the CSV
    if os.path.exists(prediction_path):
        existing_df = pd.read_csv(prediction_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    # Save updated CSV
    updated_df.to_csv(prediction_path, index=False)
    print(f"Predictions saved successfully to {prediction_path}.")

# Main execution
if __name__ == "__main__":
    try:
        # Generate predictions
        predictions = predict_future(csv_path, model_path)

        # Save predictions to CSV
        save_predictions(predictions, prediction_path)

    except Exception as e:
        print(f"Error during prediction: {e}")
