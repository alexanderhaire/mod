"""
Deep Discovery Scan (Phase 3)
=============================

Re-scanning the entire market universe using the "Advanced Engine"
(Gradient Boosting + Macro Features).

Hypothesis: 
Earlier scans failed on Tech/Crypto because we used Random Forest (weak) 
and only Price Data (blind to rates/liquidity).
With GBM + Macro (Rates/Credit/Dollar), we might find new predictability.

Universe: 50+ Assets (Commodities, Sectors, Factors, Crypto, Forex, Regional)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# LARGE UNIVERSE (The same one we used before + more)
DISCOVERY_UNIVERSE = {
    # Commodities
    'GLD': 'Gold', 'SLV': 'Silver', 'CPER': 'Copper', 'USO': 'Oil', 'UNG': 'Nat Gas', 'DBA': 'Agriculture',
    # Sectors (US)
    'XLB': 'Materials', 'XLE': 'Energy', 'XLF': 'Financials', 'XLI': 'Industrials', 
    'XLK': 'Tech', 'XLP': 'Staples', 'XLU': 'Utilities', 'XLV': 'Health', 'XLY': 'Discretionary',
    'XLC': 'Comm Svcs', 'XBI': 'Biotech', 'ITB': 'Homebuilders', 'KBE': 'Banks',
    # Global
    'EEM': 'Emerging', 'EFA': 'Developed', 'EWJ': 'Japan', 'MCHI': 'China', 'INDA': 'India', 
    'EWZ': 'Brazil', 'RSX': 'Russia', 'VGK': 'Europe',
    # Factors
    'MTUM': 'Momentum', 'VLUE': 'Value', 'USMV': 'Min Vol', 'QUAL': 'Quality', 'SIZE': 'Size',
    # Bonds
    'JNK': 'Junk Bonds', 'LQD': 'Corp Bonds', 'TLT': '20Y Treas', 'IEF': '7-10Y Treas', 'SHY': '1-3Y Treas',
    # Real Estate
    'VNQ': 'US REITs', 'REM': 'Mortgage REITs',
    # Currencies (Inv)
    'FXE': 'Euro', 'FXA': 'Aussie', 'FXY': 'Yen', 'UUP': 'Dollar',
    # Crypto Proxies
    'BITO': 'Bitcoin ETF', 'COIN': 'Coinbase', 'MSTR': 'MicroStrat'
}

MACRO_ASSETS = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    tickers = list(DISCOVERY_UNIVERSE.keys()) + MACRO_ASSETS
    tickers = list(set(tickers))
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        return prices.dropna(how='all').ffill()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_full_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if ticker not in prices.columns: return pd.DataFrame()
    
    df = pd.DataFrame(index=prices.index)
    
    # 1. Macro Features
    try:
        if '^TNX' in prices.columns:
            df['rate_change'] = prices['^TNX'].diff(20)
            df['rate_trend'] = prices['^TNX'] - prices['^TNX'].rolling(60).mean()
        if 'JNK' in prices.columns and 'IEF' in prices.columns:
            df['credit_spread'] = prices['JNK'] / prices['IEF']
        if 'UUP' in prices.columns:
            df['dollar_vol'] = prices['UUP'].pct_change().rolling(20).std()
    except:
        pass # missing macro is ok, will dropna later
        
    # 2. Asset Features
    r = prices[ticker].pct_change()
    df['mom_1m'] = r.rolling(20).mean()
    df['mom_3m'] = r.rolling(60).mean()
    df['vol_1m'] = r.rolling(20).std()
    
    # Target: 5-Day Forward Return
    df['target'] = r.shift(-5).rolling(5).sum()
    
    return df.dropna()

def test_asset(prices: pd.DataFrame, ticker: str) -> dict:
    data = create_full_features(prices, ticker)
    if len(data) < 250:
        return {'ticker': ticker, 'ic': -999, 'accuracy': 0}
        
    split = int(len(data) * 0.7)
    train, test = data.iloc[:split], data.iloc[split:]
    
    features = [c for c in data.columns if c != 'target']
    
    # THE UPGRADE: Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[features], train['target'])
    preds = model.predict(test[features])
    
    ic, _ = spearmanr(preds, test['target'])
    accuracy = accuracy_score(np.sign(test['target']), np.sign(preds))
    
    return {
        'ticker': ticker,
        'name': DISCOVERY_UNIVERSE.get(ticker, ticker),
        'ic': ic,
        'accuracy': accuracy
    }

def run_discovery():
    print("=" * 80)
    print("   DEEP DISCOVERY SCAN (GBM + MACRO)")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    print(f"   Data Shape: {prices.shape}")
    
    results = []
    
    for ticker in DISCOVERY_UNIVERSE.keys():
        try:
            res = test_asset(prices, ticker)
            if res['ic'] != -999:
                results.append(res)
        except Exception as e:
            pass
            
    # Sort by IC
    results.sort(key=lambda x: x['ic'], reverse=True)
    
    print(f"\n   {'Ticker':<8} {'Name':<20} {'IC':<8} {'Accuracy':<8}")
    print("   " + "-" * 50)
    
    # Show Top 20
    for res in results[:20]:
        print(f"   {res['ticker']:<8} {res['name']:<20} {res['ic']:<8.3f} {res['accuracy']:<8.1%}")
        
    print("\n   Notable Failures (Bottom 5):")
    for res in results[-5:]:
        print(f"   {res['ticker']:<8} {res['name']:<20} {res['ic']:<8.3f} {res['accuracy']:<8.1%}")

if __name__ == "__main__":
    run_discovery()
