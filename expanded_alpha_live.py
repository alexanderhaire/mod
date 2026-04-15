
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_live_data():
    """Fetch the last 60 days of data for the Expanded Universe."""
    tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
        'ADA-USD', 'XRP-USD', 'AVAX-USD', 'SHIB-USD', 
        'DOT-USD', 'LINK-USD', 'LTC-USD'
        # Kept only high-reliability tickers
    ]
    
    print(f"📡 Fetching live data for {len(tickers)} assets...")
    data = yf.download(tickers, period='60d', interval='1d', progress=False)
    
    # Handle yfinance columns
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                prices = data.xs('Adj Close', axis=1, level=1)
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    return prices.ffill().dropna()

def analyze_market(prices):
    """Analyze current market state and return the Buy List."""
    
    # 1. Parameter Setup
    LOOKBACK = 14
    TOP_N = 3
    
    universe = [c for c in prices.columns if c != 'BTC-USD']
    if 'BTC-USD' not in prices.columns:
        print("❌ Critical Error: BTC-USD data missing.")
        return
        
    # Get latest complete data (yesterday close if today is active)
    # We'll just take the last available row
    latest_idx = prices.index[-1]
    prev_idx = prices.index[-1-LOOKBACK]
    
    print(f"\n📅 Analysis Date: {latest_idx.date()}")
    print("-" * 50)
    
    # 2. Calculate Momentum (14d)
    mom = {}
    
    # BTC Momentum
    btc_current = prices['BTC-USD'].iloc[-1]
    btc_prev = prices['BTC-USD'].iloc[-1-LOOKBACK]
    btc_mom = btc_current / btc_prev - 1
    
    print(f"BTC Momentum (14d): {btc_mom:.2%}")
    
    # Alt Momentum
    print("\n🔍 Altcoin Momentum Scan:")
    for coin in universe:
        try:
            curr = prices[coin].iloc[-1]
            prev = prices[coin].iloc[-1-LOOKBACK]
            if prev > 0:
                m = curr / prev - 1
                mom[coin] = m
                print(f"   {coin:<10} {m:.2%}")
        except:
            pass
            
    # 3. Sort and Select Top N
    sorted_alts = sorted(mom.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_alts[:TOP_N]
    
    # 4. Compare vs BTC
    if not top_candidates:
        print("⚠️ No valid alt data found.")
        return
        
    avg_alt_mom = np.mean([x[1] for x in top_candidates])
    
    print("-" * 50)
    print(f"Top {TOP_N} Alts Avg Mom: {avg_alt_mom:.2%}")
    print(f"Bitcoin Momentum:    {btc_mom:.2%}")
    
    print("-" * 50)
    print("📢 TRADE SIGNAL:")
    
    if avg_alt_mom > btc_mom:
        print(f"🚀 ATLSEASON DETECTED! (Alts > BTC)")
        print("\n🛒 BUY LIST (Equal Weight):")
        for coin, m in top_candidates:
            print(f"   1. {coin} (Mom: {m:.1%})")
    else:
        print(f"🛡️ BITCOIN SEASON DETECTED. (BTC > Alts)")
        print("\n🛒 BUY LIST:")
        print(f"   1. BTC-USD (100% Allocation)")
        
    print("-" * 50)

if __name__ == "__main__":
    try:
        prices = fetch_live_data()
        if not prices.empty:
            analyze_market(prices)
        else:
            print("Error: No data returned from API.")
    except Exception as e:
        print(f"An error occurred: {e}")
