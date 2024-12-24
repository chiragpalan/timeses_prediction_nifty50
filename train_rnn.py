import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Load and preprocess data
def preprocess_data(file_path, lookback=60):
    # Load data
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Select Close price for training
    data = df[['Close']].values

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare input-output sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# Build RNN model
def build_rnn(lookback, dropout_rate, units):
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save the model
def train_and_save_model(file_path, model_path):
    # Preprocess data
    lookback = 60
    X, y, scaler = preprocess_data(file_path, lookback)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = build_rnn(lookback, dropout_rate=0.2, units=50)

    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save model and scaler
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, "rnn_model.h5"))
    np.save(os.path.join(model_path, "scaler.npy"), scaler)
    print("Model and scaler saved successfully.")

# Main script
if __name__ == "__main__":
    csv_path = "data/nifty50_data.csv"
    model_path = "RNN_model"
    train_and_save_model(csv_path, model_path)
