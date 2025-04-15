import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import os

def load_data(file_path):
    """
    Load CSV data, check columns, sort by date, and clean volume.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)  # Sort by date for correct calculations
        if df['volume'].dtype == 'object':
            df['volume'] = df['volume'].str.replace(',', '').astype(float)
        if len(df) < 20:
            print(f"Warning: {file_path} has only {len(df)} rows, needs 20+ for VWMA20.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_vwma(df, window=20):
    """
    Calculate Volume Weighted Moving Average manually.
    """
    try:
        if len(df) < window:
            print(f"Cannot calculate VWMA{window}: Not enough data ({len(df)} rows).")
            return None
        df['close_volume'] = df['close'] * df['volume']
        df['vwma_num'] = df['close_volume'].rolling(window=window).sum()
        df['vwma_den'] = df['volume'].rolling(window=window).sum()
        df['vwma20'] = df['vwma_num'] / df['vwma_den']
        df = df.drop(['close_volume', 'vwma_num', 'vwma_den'], axis=1)
        return df
    except Exception as e:
        print(f"Error calculating VWMA: {e}")
        return None

def calculate_indicators(df):
    """
    Calculate EMA10, EMA20, VWMA20, and RSI14 indicators.
    """
    try:
        ema10 = EMAIndicator(close=df['close'], window=10)
        df['ema10'] = ema10.ema_indicator()
        ema20 = EMAIndicator(close=df['close'], window=20)
        df['ema20'] = ema20.ema_indicator()
        df = calculate_vwma(df, window=20)
        if df is None:
            return None
        rsi14 = RSIIndicator(close=df['close'], window=14)
        df['rsi14'] = rsi14.rsi()
        return df
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

def generate_signals(df):
    """
    Generate buy/sell signals using only EMA crosses to ensure trades.
    """
    try:
        df['signal'] = 0  # 0 = no action, 1 = buy, -1 = sell
        df['buy_condition'] = (df['ema10'] > df['vwma20']) & (df['ema10'].shift(1) <= df['vwma20'].shift(1))
        df.loc[df['buy_condition'], 'signal'] = 1  # No RSI condition
        df['sell_condition'] = (df['ema10'] < df['ema20']) & (df['ema10'].shift(1) >= df['ema20'].shift(1))
        df.loc[df['sell_condition'], 'signal'] = -1  # No RSI condition
        df = df.drop(['buy_condition', 'sell_condition'], axis=1)
        if df['signal'].eq(1).sum() == 0:
            print("No buy signals. Check if EMA10 crosses VWMA20.")
        if df['signal'].eq(-1).sum() == 0:
            print("No sell signals. Check if EMA10 crosses EMA20.")
        # Save RSI14 and EMA values for debugging
        df[['rsi14', 'ema10', 'vwma20', 'ema20']].to_csv("indicator_values.csv")
        print("RSI14, EMA10, VWMA20, EMA20 saved to indicator_values.csv.")
        return df
    except Exception as e:
        print(f"Error generating signals: {e}")
        return None

def simulate_trades(df, initial_capital=10000, transaction_fee=2):
    """
    Simulate trades with stop-loss (5%) and take-profit (10%).
    """
    try:
        capital = initial_capital
        shares = 0
        entry_price = 0
        trades = []
        in_position = False

        for date, row in df.iterrows():
            price = row['close']
            signal = row['signal']

            if in_position:
                stop_loss_price = entry_price * 0.95
                take_profit_price = entry_price * 1.10
                exit_reason = None
                if row['low'] <= stop_loss_price:
                    exit_price = min(price, stop_loss_price)
                    exit_reason = 'stop_loss'
                elif row['high'] >= take_profit_price:
                    exit_price = max(price, take_profit_price)
                    exit_reason = 'take_profit'
                elif signal == -1:
                    exit_price = price
                    exit_reason = 'sell_signal'

                if exit_reason:
                    revenue = shares * exit_price - transaction_fee
                    profit_loss = revenue - (shares * entry_price + transaction_fee)
                    capital += revenue
                    trades.append({
                        'entry_time': entry_date,
                        'exit_time': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'profit_loss': profit_loss,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'exit_reason': exit_reason,
                        'capital': capital
                    })
                    shares = 0
                    in_position = False

            if signal == 1 and capital > 0 and not in_position:
                shares_to_buy = (capital - transaction_fee) // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price + transaction_fee
                    capital -= cost
                    shares = shares_to_buy
                    entry_price = price
                    entry_date = date
                    in_position = True
                    trades.append({
                        'entry_time': entry_date,
                        'exit_time': None,
                        'entry_price': entry_price,
                        'exit_price': None,
                        'shares': shares,
                        'profit_loss': 0,
                        'stop_loss': entry_price * 0.95,
                        'take_profit': entry_price * 1.10,
                        'exit_reason': 'buy',
                        'capital': capital
                    })

        if in_position:
            exit_price = df['close'].iloc[-1]
            revenue = shares * exit_price - transaction_fee
            profit_loss = revenue - (shares * entry_price + transaction_fee)
            capital += revenue
            trades.append({
                'entry_time': entry_date,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'profit_loss': profit_loss,
                'stop_loss': entry_price * 0.95,
                'take_profit': entry_price * 1.10,
                'exit_reason': 'end_of_data',
                'capital': capital
            })

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No trades generated. Check signals in data_with_indicators_and_signals.csv.")
        return trades_df, capital, shares
    except Exception as e:
        print(f"Error simulating trades: {e}")
        return None, None, None

def main():
    file_path = "TSLA.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found in the project folder.")
        return
    df = load_data(file_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    df = calculate_indicators(df)
    if df is None:
        print("Failed to calculate indicators. Exiting.")
        return
    df = generate_signals(df)
    if df is None:
        print("Failed to generate signals. Exiting.")
        return
    trades_df, final_capital, final_shares = simulate_trades(df)
    if trades_df is None:
        print("Failed to simulate trades. Exiting.")
        return
    df.to_csv("data_with_indicators_and_signals_TSLA.csv")
    trades_df.to_csv("trades.csv")
    print(f"Data with indicators and signals saved to data_with_indicators_and_signals.csv")
    print(f"Trades saved to trades.csv")
    print(f"Final capital: {final_capital:.2f}, Final shares: {final_shares}")

if __name__ == "__main__":
    main()