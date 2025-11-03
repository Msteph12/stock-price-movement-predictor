# src/downloaddata.py
import yfinance as yf
from pathlib import Path
import pandas as pd
from datetime import datetime

def download_stock_data(
    ticker: str = "AAPL",
    start_date: str = "2025-01-01",
    end_date: str = "2025-08-18",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Downloads stock data from Yahoo Finance and saves it as CSV.
    """
    # Create data/raw directory if it doesn't exist
    raw_dir = Path(__file__).resolve().parents[1] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    # ✅ PROPERLY Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Extract just the price column names (Open, High, Low, Close, Volume)
        df.columns = df.columns.droplevel(1)  # Remove the ticker level
    
    # Save to CSV
    filename = f"{ticker}_{start_date}_{end_date}_daily.csv"
    filepath = raw_dir / filename
    df.to_csv(filepath)
    print(f"✅ Data saved to {filepath}")
    
    return df

if __name__ == "__main__":
    download_stock_data()
    