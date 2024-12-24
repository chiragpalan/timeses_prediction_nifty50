import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# File paths
csv_path = "data/nifty50_data.csv"  # Input CSV file
model_save_path = "model/rnn_model"  # Folder to save the trained model

# Function to load and preprocess data
def load_data(csv_path, lookback=60):
    data = pd.read_csv(csv_path)
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled_features = (features - features.mean(axis=0)) / features.std(axis=0)

    # Prepare data for RNN
    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i - lookback:i])  # Lookback window
        y.append(scaled_features[i, 3])  # Predict 'Close' price

    return np.array(X), np.array(y)

# Function to create and train RNN model
def train_rnn(X, y):
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32)
    return model

# Save the trained model
def save_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved successfully at {save_path}.")

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist.")
    X, y = load_data(csv_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RNN model
    print("Training RNN model...")
    rnn_model = train_rnn(X_train, y_train)

    # Save the trained model
    save_model(rnn_model, model_save_path)
