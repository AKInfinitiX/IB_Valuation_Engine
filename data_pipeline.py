import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers, start_date, end_date):
    print(f"Downloading data for: {tickers}...")
    
    # Download data with auto_adjust to directly obtain split/dividend-adjusted close
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # Handle single or multi-index column structures safely
    if 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        data = raw_data
        
    data = data.ffill().dropna()
    log_returns = np.log(data / data.shift(1)).dropna()
    
    print(f"Data fetched successfully. Shape: {log_returns.shape}")
    return data, log_returns

if __name__ == "__main__":
    portfolio_tickers = ['AAPL', 'SPY', 'GLD', 'TLT']
    prices, returns = fetch_data(tickers=portfolio_tickers, start_date="2016-01-01", end_date="2026-01-01")
    print(returns.head(3))