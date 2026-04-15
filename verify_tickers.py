
import yfinance as yf
import pandas as pd

tickers = [
    'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', 'UUP', 'XLE', # Trad
    'VUG', 'VTV', 'RSP', # Factors/Breadth
    '^FVX', '^TYX', '^VIX', # Macro
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD' # Crypto
]

print(f"Testing download for {len(tickers)} tickers...")
try:
    data = yf.download(tickers, start='2024-01-01', progress=True)
    
    if data.empty:
        print("CRITICAL: Download returned empty DataFrame!")
    else:
        print("Download successful.")
        # Check individual columns
        # yfinance returns MultiIndex (Price, Ticker) or just (Ticker) depending on version/args
        # With multiple tickers, it's usually (Price, Ticker)
        
        found_tickers = []
        if isinstance(data.columns, pd.MultiIndex):
            # Level 1 is likely Ticker
            found_tickers = data.columns.get_level_values(1).unique().tolist()
            if not found_tickers:
                 # Maybe level 0?
                 found_tickers = data.columns.get_level_values(0).unique().tolist()
        else:
            found_tickers = data.columns.tolist()
            
        print(f"Found columns for: {found_tickers}")
        
        missing = set(tickers) - set(found_tickers)
        if missing:
            print(f"MISSING DATA FOR: {missing}")
            # Try individual downloads for missing
            for t in missing:
                print(f"Attempting individual fetch for {t}...")
                s = yf.download(t, start='2024-01-01', progress=False)
                if s.empty:
                    print(f"  FAILED: {t} is truly unretrievable.")
                else:
                    print(f"  SUCCESS: {t} works individually.")
        else:
            print("All tickers have data columns.")

except Exception as e:
    print(f"Exception during download: {e}")
