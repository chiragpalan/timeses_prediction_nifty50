import os
import yfinance as yf
import pandas as pd

# Define the CSV file path
csv_file_path = "data/nifty50_data.csv"

# Function to download Nifty 50 data
def download_nifty50():
    print("Downloading Nifty 50 data...")
    ticker = "^NSEI"
    data = yf.download(ticker, interval="5m", period="1d")
    
    # Reset index to make 'Datetime' a column and ensure the format
    data.reset_index(inplace=True)
    
    # Ensure 'Datetime' column is in the proper string format
    data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Return the data
    return data

# Function to save or append data to a CSV file
def save_to_csv(data, csv_file_path):
    if not os.path.exists(csv_file_path):
        print("CSV file does not exist. Creating a new file...")
        data.to_csv(csv_file_path, index=False)
        print("Data saved to new CSV file successfully.")
    else:
        print("CSV file exists. Appending data...")
        # Append to the existing file
        data.to_csv(csv_file_path, mode='a', header=False, index=False)
        print("Data appended successfully.")

# Main execution
if __name__ == "__main__":
    # Download the data
    nifty_data = download_nifty50()
    
    # Save or append the data to the CSV file
    save_to_csv(nifty_data, csv_file_path)
