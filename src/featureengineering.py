# src/featureengineering.py 
import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands.
    """
    df = df.copy()
    
    # ✅ Ensure 'Close' is a 1D Series
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].squeeze()
    
    # Create a guaranteed 1D close series
    close_values = np.array(df["Close"]).flatten()  # This will always be 1D
    close_series = pd.Series(close_values, index=df.index, name="Close")
    
    print(f"Close series shape: {close_series.shape}")
    print(f"Close series type: {type(close_series)}")

    # --- Moving Averages ---
    df["SMA_5"] = close_series.rolling(window=5).mean()
    df["SMA_10"] = close_series.rolling(window=10).mean()
    df["EMA_5"] = close_series.ewm(span=5, adjust=False).mean()
    df["EMA_10"] = close_series.ewm(span=10, adjust=False).mean()

    # --- Momentum Indicators ---
    # Use the guaranteed 1D close_series for ALL ta indicators
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
    
    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # --- Volatility Indicators ---
    bb = ta.volatility.BollingerBands(close=close_series)
    df["Bollinger_high"] = bb.bollinger_hband()
    df["Bollinger_low"] = bb.bollinger_lband()

    # --- Daily Return ---
    df["Return"] = close_series.pct_change()

    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features matrix X and target vector y from raw dataframe."""
    df = df.copy()

    # If Date is in index, move it into a column
    if df.index.name == 'Date' or 'Date' not in df.columns:
        try:
            df = df.reset_index()
        except Exception:
            pass

    # Ensure Close exists and is 1D
    if 'Close' not in df.columns:
        raise KeyError("'Close' column is required in the input dataframe")
    
    # Force Close to be 1D
    if hasattr(df['Close'], 'shape') and len(df['Close'].shape) > 1:
        df['Close'] = df['Close'].squeeze()

    # Add technical indicators
    df = add_technical_indicators(df)

    # Drop rows with NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)

    # Create target: next day's close > today's close
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df[:-1]  # drop last row which has no next-day label

    # Choose features (exclude Date and Target and any non-numeric columns)
    exclude = {'Date', 'Target'}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].copy()
    y = df['Target'].copy()

    return X, y