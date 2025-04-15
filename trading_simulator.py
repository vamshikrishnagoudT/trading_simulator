import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import os

def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame and check for required columns.
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower().str.strip()
        # Check for required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        # Clean volume: remove commas and convert to numeric
        if df['volume'].dtype == 'object':
            df['volume'] = df['volume'].str.replace(',', '').astype(float)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_vwma(df, window=20):
    """
    Calculate Volume Weighted Moving Average manually.
    VWMA = Sum(Close * Volume) / Sum(Volume) over the window.
    """
    try:
        # Calculate Close * Volume
        df['close_volume'] = df['close'] * df['volume']
        # Calculate rolling sum of Close * Volume and Volume
        df['vwma_num'] = df['close_volume'].rolling(window=window).sum()
        df['vwma_den'] = df['volume'].rolling(window=window).sum()
        # Calculate VWMA
        df['vwma20'] = df['vwma_num'] / df['vwma_den']
        # Drop temporary columns
        df = df.drop(['close_volume', 'vwma_num', 'vwma_den'], axis=1)
        return df
    except Exception as e:
        print(f"Error calculating VWMA: {e}")
        return None

def calculate_indicators(df):
    """
    Calculate EMA10, VWMA20, and RSI14 indicators.
    """
    try:
        # Calculate EMA10
        ema10 = EMAIndicator(close=df['close'], window=10)
        df['ema10'] = ema10.ema_indicator()

        # Calculate VWMA20 manually
        df = calculate_vwma(df, window=20)
        if df is None:
            return None

        # Calculate RSI14
        rsi14 = RSIIndicator(close=df['close'], window=14)
        df['rsi14'] = rsi14.rsi()

        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

def main():
    # Define file path
    file_path = "TSLA.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found in the project folder.")
        return
    
    # Load data
    df = load_data(file_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Calculate indicators
    df = calculate_indicators(df)
    if df is None:
        print("Failed to calculate indicators. Exiting.")
        return
    
    # Save results to a new CSV
    output_path = "data_with_indicators_TSLA.csv"
    df.to_csv(output_path)
    print(f"Data with indicators saved to {output_path}")

if __name__ == "__main__":
    main()