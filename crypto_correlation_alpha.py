"""
Crypto-Prediction Correlation Engine
====================================

Finds the "Lag" between Real Asset Prices (BTC, DOGE) and Prediction Markets.
Hypothesis: Prediction markets are slow to update after sudden crypto price moves.

Steps:
1. Fetch 7d hourly data for BTC-USD, DOGE-USD (yfinance).
2. Load Polymarket history.
3. Identify Crypto-Related Markets via keywords.
4. Calculate Correlation and Lag.

"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_poly_data():
    try:
        df = pd.read_csv('data/polymarket_real.csv')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        return df
    except:
        return None

def fetch_crypto_data(symbol="BTC-USD"):
    print(f"   Fetching {symbol} from Yahoo Finance...")
    try:
        # Fetch last 7 days hourly
        df = yf.download(symbol, period="7d", interval="1h", progress=False)
        if len(df) == 0: return None
        # Keep Close
        series = df['Close']
        # Remove timezone if needed or ensure UTC
        if series.index.tz is None:
            # Assume UTC if none, or localize
            pass
        else:
             series.index = series.index.tz_convert(None)
             
        # Resample to align with Poly (flooring to hour)
        series = series.resample('1h').last().ffill()
        return series
    except Exception as e:
        print(f"      Error fetching yf: {e}")
        return None

def main():
    print("₿ CRYPTO-PREDICTION CORRELATION")
    print("===============================")
    
    poly_df = load_poly_data()
    if poly_df is None: return
    
    # 1. Identify Crypto Markets
    # Keywords map to Tickers
    assets = {
        "Bitcoin": "BTC-USD",
        "BTC": "BTC-USD",
        "Ethereum": "ETH-USD",
        "ETH": "ETH-USD",
        "DOGE": "DOGE-USD",
        "Solana": "SOL-USD",
        "SOL": "SOL-USD"
    }
    
    print("\n1️⃣  Fetching Real Crypto Prices...")
    crypto_prices = {}
    for kw, ticker in assets.items():
        if ticker not in crypto_prices:
            data = fetch_crypto_data(ticker)
            if data is not None:
                crypto_prices[ticker] = data
                
    print(f"   Loaded {len(crypto_prices)} crypto assets.")
    
    # 2. Correlate
    print("\n2️⃣  Scanning for Correlations...")
    print(f"{'Market Event (Short)':<40} {'Asset':<8} {'Corr':<6} {'Lag Effect'}")
    print("-" * 100)
    
    # Pivot Poly Data
    price_matrix = poly_df.pivot_table(index='datetime', columns='Question', values='Price')
    price_matrix = price_matrix.resample('1h').last().ffill()
    
    matches = 0
    
    # Iterate all Poly Markets
    for question in price_matrix.columns:
        # Check if question mentions an asset
        target_ticker = None
        for kw, ticker in assets.items():
            if kw.lower() in question.lower():
                target_ticker = ticker
                break
        
        if not target_ticker:
            continue
            
        if target_ticker not in crypto_prices:
            continue
            
        # Align Data
        poly_series = price_matrix[question]
        asset_series = crypto_prices[target_ticker]
        
        # Squeeze dimensions if necessary (YF returns dataframe often)
        if isinstance(asset_series, pd.DataFrame):
            asset_series = asset_series.iloc[:, 0]
            
        # Common Index
        common = poly_series.index.intersection(asset_series.index)
        if len(common) < 10:
            continue
            
        p_price = poly_series.loc[common]
        a_price = asset_series.loc[common]
        
        # Calculate Correlation
        corr = p_price.corr(a_price)
        
        matches += 1
        
        # Interpret
        lag_status = "⚪ Sync"
        if abs(corr) > 0.8:
            lag_status = "🟢 Coupled"
        elif abs(corr) > 0.4:
            lag_status = "🟡 Loose Link"
        else:
            lag_status = "🔴 Broken Link (Alpha?)"
            
        label = f"{question[:35]}..."
        print(f"{label:<40} {target_ticker:<8} {corr:<6.2f} {lag_status}")
        
    if matches == 0:
        print("   No crypto-related markets found in current data.")

if __name__ == "__main__":
    main()
