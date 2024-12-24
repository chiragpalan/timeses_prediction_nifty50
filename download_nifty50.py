import pandas as pd
from datetime import datetime
import sqlite3
import os
from git import Repo
import yfinance as yf

# Function to download Nifty 50 data
def download_nifty50_data():
    ticker = "^NSEI"  # Ticker for Nifty 50
    data = yf.download(ticker, interval="5m", period="1d")
    data.reset_index(inplace=True)
    return data

# Append data to database
def append_to_database(data, db_path):
    conn = sqlite3.connect(db_path)
    data.to_sql("nifty50", conn, if_exists="append", index=False)
    conn.close()

# Push changes to GitHub
def push_to_github(repo_path, commit_message):
    repo = Repo(repo_path)
    repo.git.add(update=True)
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    origin.push()

if __name__ == "__main__":
    # Define file paths
    repo_path = "/github/workspace/timeses_prediction_nifty50"  # GitHub repo path
    db_path = os.path.join(repo_path, "data", "nifty50.db")

    # Download data
    print("Downloading Nifty 50 data...")
    nifty_data = download_nifty50_data()

    # Append to database
    print("Appending data to database...")
    append_to_database(nifty_data, db_path)

    # Push changes to GitHub
    print("Pushing changes to GitHub...")
    push_to_github(repo_path, f"Daily update: {datetime.now().strftime('%Y-%m-%d')}")

    print("Task completed!")

