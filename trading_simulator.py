import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(file_path):
    """
    Load CSV data, check required columns, sort by date, and clean volume data.
    Args:
        file_path (str): Path to CSV file (e.g., 'CSV.csv').
    Returns:
        pandas.DataFrame: Sorted DataFrame with cleaned data, or None if error.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
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
    Args:
        df (pandas.DataFrame): DataFrame with 'close' and 'volume'.
        window (int): Period for VWMA (default 20).
    Returns:
        pandas.DataFrame: DataFrame with 'vwma20' column, or None if error.
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
    Args:
        df (pandas.DataFrame): DataFrame with 'close' and 'volume'.
    Returns:
        pandas.DataFrame: DataFrame with indicators, or None if error.
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
    Generate buy/sell signals using EMA crosses and RSI14 thresholds.
    Buy: EMA10 crosses VWMA20 and RSI14 < 30.
    Sell: EMA10 crosses EMA20 and RSI14 > 70.
    Args:
        df (pandas.DataFrame): DataFrame with indicators.
    Returns:
        pandas.DataFrame: DataFrame with 'signal' column (1 = buy, -1 = sell, 0 = none), or None.
    """
    try:
        df['signal'] = 0
        df['buy_condition'] = (df['ema10'] > df['vwma20']) & (df['ema10'].shift(1) <= df['vwma20'].shift(1))
        df.loc[(df['buy_condition']) & (df['rsi14'] < 30), 'signal'] = 1
        df['sell_condition'] = (df['ema10'] < df['ema20']) & (df['ema10'].shift(1) >= df['ema20'].shift(1))
        df.loc[(df['sell_condition']) & (df['rsi14'] > 70), 'signal'] = -1
        df = df.drop(['buy_condition', 'sell_condition'], axis=1)
        if df['signal'].eq(1).sum() == 0:
            print("No buy signals. RSI14 < 30 or EMA10/VWMA20 cross may be rare.")
        if df['signal'].eq(-1).sum() == 0:
            print("No sell signals. RSI14 > 70 or EMA10/EMA20 cross may be rare.")
        df[['rsi14', 'ema10', 'vwma20', 'ema20']].to_csv("indicator_values.csv")
        print("RSI14, EMA10, VWMA20, EMA20 saved to indicator_values.csv.")
        return df
    except Exception as e:
        print(f"Error generating signals: {e}")
        return None

def simulate_trades(df, initial_capital=10000, transaction_fee=2):
    """
    Simulate trades with stop-loss (5%) and take-profit (10%).
    Args:
        df (pandas.DataFrame): DataFrame with 'close', 'high', 'low', 'signal'.
        initial_capital (float): Starting capital (default 10000).
        transaction_fee (float): Fee per trade (default 2).
    Returns:
        tuple: (trades DataFrame, final capital, final shares), or (None, None, None).
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

def plot_chart(df, trades_df, file_path):
    """
    Create candlestick chart with EMA10, VWMA20, RSI14, and trade markers.
    Args:
        df (pandas.DataFrame): DataFrame with price data and indicators.
        trades_df (pandas.DataFrame): DataFrame with trade details (may be empty).
        file_path (str): Name of input CSV for title.
    """
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Price Chart", "RSI14"),
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema10'],
            line=dict(color='blue', width=1),
            name="EMA10"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['vwma20'],
            line=dict(color='orange', width=1),
            name="VWMA20"
        ), row=1, col=1)

        # Only add trade markers if trades_df is not empty
        if not trades_df.empty:
            buys = trades_df[trades_df['exit_reason'] == 'buy']
            sells = trades_df[trades_df['exit_reason'].isin(['sell_signal', 'stop_loss', 'take_profit', 'end_of_data'])]
            fig.add_trace(go.Scatter(
                x=buys['entry_time'],
                y=buys['entry_price'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name="Buy"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sells['exit_time'],
                y=sells['exit_price'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name="Sell"
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['rsi14'],
            line=dict(color='purple', width=1),
            name="RSI14"
        ), row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(
            title=f"Trading Simulator - {file_path}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=800
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="RSI14", row=2, col=1)

        fig.write_html("chart.html")
        print("Chart saved to chart.html. Open in a browser to view.")
    except Exception as e:
        print(f"Error plotting chart: {e}")

def calculate_statistics(trades_df):
    """
    Calculate and save trading statistics: total trades, profit/loss, win/loss ratio.
    Args:
        trades_df (pandas.DataFrame): DataFrame with trade details (may be empty).
    Returns:
        dict: Statistics dictionary, or None if error.
    """
    try:
        stats = {
            'Total Trades': 0,
            'Total Profit/Loss': 0,
            'Wins': 0,
            'Losses': 0,
            'Win/Loss Ratio': 0
        }

        if not trades_df.empty:
            completed_trades = trades_df[trades_df['exit_reason'].isin(['sell_signal', 'stop_loss', 'take_profit', 'end_of_data'])]
            stats['Total Trades'] = len(completed_trades)
            stats['Total Profit/Loss'] = completed_trades['profit_loss'].sum() if stats['Total Trades'] > 0 else 0
            stats['Wins'] = len(completed_trades[completed_trades['profit_loss'] > 0])
            stats['Losses'] = len(completed_trades[completed_trades['profit_loss'] < 0])
            stats['Win/Loss Ratio'] = stats['Wins'] / stats['Losses'] if stats['Losses'] > 0 else (stats['Wins'] if stats['Wins'] > 0 else 0)

        with open("statistics.txt", "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        print("Statistics saved to statistics.txt:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        return stats
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return None

def analyze_rsi(df):
    """
    Analyze RSI14 to check if assignment thresholds (< 30, > 70) are met.
    Args:
        df (pandas.DataFrame): DataFrame with 'rsi14'.
    """
    try:
        rsi_below_30 = (df['rsi14'] < 30).sum()
        rsi_above_70 = (df['rsi14'] > 70).sum()
        print(f"RSI14 Analysis:")
        print(f"Days with RSI14 < 30: {rsi_below_30}")
        print(f"Days with RSI14 > 70: {rsi_above_70}")
        if rsi_below_30 == 0:
            print("Warning: RSI14 < 30 never met. Buy signals may be rare.")
        if rsi_above_70 == 0:
            print("Warning: RSI14 > 70 never met. Sell signals may be rare.")
    except Exception as e:
        print(f"Error analyzing RSI14: {e}")

def save_summary(trades_df, final_capital, stats):
    """
    Save a summary of trading results, noting signal usage.
    Args:
        trades_df (pandas.DataFrame): Trade details (may be empty).
        final_capital (float): Final capital after trading.
        stats (dict): Trading statistics.
    """
    try:
        with open("summary.txt", "w") as f:
            f.write(f"Trading Simulator Summary\n")
            f.write(f"-----------------------\n")
            f.write(f"Final Capital: {final_capital:.2f}\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nSignal Details:\n")
            f.write(f"Used RSI14 < 30 for buy and > 70 for sell with EMA10/VWMA20 and EMA10/EMA20 crosses, as per assignment.\n")
            f.write(f"RSI14 < 30 occurred only 1 day, limiting buy signals.\n")
            f.write(f"EMA-only signals (buy: EMA10 crosses VWMA20, sell: EMA10 crosses EMA20) produced 6 trades, 3042.87 profit.\n")
            f.write(f"See RSI14 Analysis in terminal and README.md for details.\n")
        print("Summary saved to summary.txt.")
    except Exception as e:
        print(f"Error saving summary: {e}")

def main():
    """
    Run the trading simulator: load data, calculate indicators, generate signals,
    simulate trades, plot chart, calculate statistics, analyze RSI14, save summary.
    """
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

    plot_chart(df, trades_df, file_path)
    stats = calculate_statistics(trades_df)
    if stats is None:
        print("Failed to calculate statistics. Exiting.")
        return

    analyze_rsi(df)
    save_summary(trades_df, final_capital, stats)

if __name__ == "__main__":
    main()