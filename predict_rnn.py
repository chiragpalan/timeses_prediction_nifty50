import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model and scaler
def load_model_and_scaler(model_path):
    model = load_model(os.path.join(model_path, "rnn_model.h5"))
    scaler = np.load(os.path.join(model_path, "scaler.npy"), allow_pickle=True).item()
    return model, scaler

# Predict future values
def predict_future(file_path, model_path, lookahead=[15, 30, 45, 60]):
    # Load data and preprocess
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Prepare data for prediction
    data = df[['Close']].values
    model, scaler = load_model_and_scaler(model_path)
    scaled_data = scaler.transform(data)

    # Use last `lookback` data points for prediction
    lookback = 60
    input_seq = scaled_data[-lookback:].reshape(1, lookback, 1)

    # Predict next values
    predictions = []
    for minutes in lookahead:
        pred = model.predict(input_seq)
        predictions.append(scaler.inverse_transform(pred)[0][0])
        # Append prediction to sequence for next prediction
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

    return predictions

# Main script
if __name__ == "__main__":
    csv_path = "data/nifty50_data.csv"
    model_path = "RNN_model"
    predictions = predict_future(csv_path, model_path)
    
    print("Predictions (in minutes):")
    print(f"15 mins: {predictions[0]}")
    print(f"30 mins: {predictions[1]}")
    print(f"45 mins: {predictions[2]}")
    print(f"60 mins: {predictions[3]}")
