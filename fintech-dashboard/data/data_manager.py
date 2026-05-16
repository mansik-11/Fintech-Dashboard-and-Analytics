import yfinance as yf
import pandas as pd
from datetime import datetime
import os

TICKER = "MSFT"
START_DATE = "2018-01-01"

def fetch_and_save_data(ticker=TICKER, start=START_DATE, data_dir=None):
    """
    Fetches historical stock data from Yahoo Finance, cleans it, and saves it to a CSV file.
    """
    if data_dir is None:
        # If no directory is provided, save in the same directory as the script.
        data_dir = os.path.dirname(os.path.abspath(__file__))

    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end_date)
    
    data.dropna(inplace=True)
    
    file_path = os.path.join(data_dir, f'{ticker}_historical.csv')
    data.to_csv(file_path)
    print(f"Data for {ticker} from {start} to {end_date} saved to {file_path}")
    return data

if __name__ == '__main__':
    # When run directly, save in the script's directory.
    fetch_and_save_data()
