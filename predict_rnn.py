# Ensure the data folder exists
os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

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
