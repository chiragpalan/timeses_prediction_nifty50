import sqlite3
import yfinance as yf
import os
import pandas as pd

# Define database path
db_path = "data/nifty50.db"

# Function to create a database if it doesn't exist and create table dynamically
def initialize_database(db_path, data):
    if not os.path.exists(db_path):
        print("Database does not exist. Creating a new database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Dynamically create the table schema based on the DataFrame columns
        columns = data.columns
        column_definitions = []
        
        for col in columns:
            # Determine column type (TEXT for Datetime, REAL for numerical data)
            if data[col].dtype == 'object':  # For Datetime or other non-numerical columns
                column_definitions.append(f'"{col}" TEXT')
            else:  # For numerical columns
                column_definitions.append(f'"{col}" REAL')

        # Create table with dynamic column definitions
        create_table_query = f"CREATE TABLE IF NOT EXISTS nifty50 ({', '.join(column_definitions)});"
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()
        print("Database and table created successfully.")
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
    # Download the data
    nifty_data = download_nifty50()

    # Ensure the database exists and the table schema is created dynamically
    initialize_database(db_path, nifty_data)

    # Append data to the database
    print("Appending data to database...")
    append_to_database(nifty_data, db_path)
