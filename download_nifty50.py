import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime

# Define database path
db_path = "data/nifty50.db"

# Function to create a database if it doesn't exist
def initialize_database(db_path):
    if not os.path.exists(db_path):
        print("Database does not exist. Creating a new database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create a table to store Nifty 50 data with the exact columns from yfinance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty50 (
                Datetime TEXT PRIMARY KEY,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        print("Database created successfully.")
    else:
        print("Database already exists.")

# Function to append data to the database
def append_to_database(data, db_path):
    conn = sqlite3.connect(db_path)
    try:
        # Append the data as is, without renaming columns
        data.to_sql('nifty50', conn, if_exists='append', index=False)
        print("Data appended successfully.")
    except Exception as e:
        print(f"Error appending data: {e}")
    finally:
        conn.close()

# Function to download Nifty 50 data
def download_nifty50():
    print("Downloading Nifty 50 data...")
    ticker = "^NSEI"
    data = yf.download(ticker, interval="5m", period="1d")
    
    # Reset index to make 'Datetime' a column and ensure the format
    data.reset_index(inplace=True)
    
    # Ensure 'Datetime' column is in the proper string format
    data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Return the data as is, with original column names
    return data

# Main execution
if __name__ == "__main__":
    # Ensure the database exists
    initialize_database(db_path)

    # Download data
    nifty_data = download_nifty50()

    # Append to database
    print("Appending data to database...")
    append_to_database(nifty_data, db_path)