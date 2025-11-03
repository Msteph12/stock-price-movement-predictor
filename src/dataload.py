# src/data_load.py
from pathlib import Path
import pandas as pd
import numpy as np
from downloaddata import download_stock_data

def load_stock_data(
    ticker: str = "AAPL",
    start_date: str = "2025-01-01",
    end_date: str = "2025-08-18",
    refresh: bool = False
) -> pd.DataFrame:
    """
    Loads stock data from CSV if available, otherwise downloads it.
    """
    filename = f"{ticker}_{start_date}_{end_date}_daily.csv"
    filepath = Path("data/raw") / filename

    if refresh or not filepath.exists():
        print("Downloading fresh data...")
        return download_stock_data(ticker, start_date, end_date)

    print("Loading data from CSV...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # ✅ If we still have MultiIndex columns from old CSV files, fix them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # ✅ Ensure Close is a 1D Series
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].squeeze()
    
    print("Close column info:")
    print(f"Type: {type(df['Close'])}")
    print(f"Shape: {df['Close'].shape if hasattr(df['Close'], 'shape') else 'N/A'}")
    print(df["Close"].head())

    return df