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
    
    Parameters:
        ticker: Stock symbol (default: 'AAPL')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval ('1d' for daily)
        
    Returns:
        DataFrame with the stock data
    """
    # Create data/raw directory if it doesn't exist
    raw_dir = Path(__file__).resolve().parents[1] / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    # Save to CSV
    filename = f"{ticker}_{start_date}_{end_date}_daily.csv"
    filepath = raw_dir / filename
    df.to_csv(filepath)
    print(f"✅ Data saved to {filepath}")
    
    return df

if __name__ == "__main__":
    # This will run when you execute the file directly
    download_stock_data()