"""
ML Discovery: Where does the Brain work?
========================================

Testing the ML ensemble on a wide variety of assets to find where
the specific Momentum/Volatility inputs have predictive power.

Hypothesis: Works better on Commodities & Macro assets than Equities.

Universe:
1. Commodities: Silver (SLV), Oil (USO), Copper (CPER), Ag (DBA)
2. Currencies: Dollar (UUP), Euro (FXE), Yen (FXY)
3. Crypto Proxies: BITO (Bitcoin), GDX (Miners), COIN
4. US Sectors: Energy (XLE), Tech (XLK), Utilities (XLU)
5. Factors: Momentum (MTUM), Value (VLUE)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DISCOVERY UNIVERSE
# =============================================================================

DISCOVERY_UNIVERSE = {
    # Commodities
    'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Oil', 'CPER': 'Copper',
    'DBA': 'Agriculture', 'GDX': 'Gold Miners', 'PALL': 'Palladium',
    'UNG': 'Natural Gas', 'CORN': 'Corn',
    
    # Global Equities
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq', 'IWM': 'Russell 2000',
    'EEM': 'Emerging Mkts', 'VGK': 'Europe', 'EWZ': 'Brazil',
    'INDA': 'India', 'FXI': 'China', 'EWJ': 'Japan',
    'EWW': 'Mexico', 'ARGT': 'Argentina', 'RSX': 'Russia (Legacy)',
    
    # US Sectors
    'XLE': 'Energy', 'XLF': 'Financials', 'XLK': 'Tech',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLP': 'Staples',
    'XLU': 'Utilities', 'XLY': 'Discretionary', 'XLB': 'Materials',
    'XLC': 'Comm Services', 'XBI': 'Biotech', 'SMH': 'Semiconductors',
    'KRE': 'Regional Banks', 'ITB': 'Homebuilders',
    
    # Factors
    'MTUM': 'Momentum', 'VLUE': 'Value', 'QUAL': 'Quality',
    'USMV': 'Min Vol', 'IJR': 'Small Cap Core',
    
    # Bonds & Credit
    'TLT': '20y Treasury', 'IEF': '7-10y Treasury', 'SHy': '1-3y Treasury',
    'LQD': 'Corp Bonds', 'HYG': 'High Yield', 'JNK': 'Junk Bonds',
    'TIP': 'TIPS', 'EMB': 'EM Bonds',
    
    # Real Estate
    'VNQ': 'Real Estate', 'REM': 'Mortgage REITs',
    
    # Currencies (Long)
    'UUP': 'US Dollar', 'FXE': 'Euro', 'FXY': 'Yen',
    'FXB': 'British Pound', 'FXA': 'Aussie Dollar',
    
    # Crypto Proxies
    'BITO': 'Bitcoin ETF', 'COIN': 'Coinbase', 'MSTR': 'MicroStrategy',
    'MARA': 'Marathon Digital'
}

# =============================================================================
# LOGIC
# =============================================================================

def fetch_data(tickers: list, years: int = 4) -> pd.DataFrame:
    """Fetch data. Using 4 years to include recent crypto proxies."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"Fetching data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # Drop columns with too much missing data
    prices = prices.dropna(axis=1, thresh=int(len(prices)*0.8))
    return prices.ffill().dropna()

def create_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create features for a single asset."""
    returns = prices[ticker].pct_change()
    df = pd.DataFrame(index=prices.index)
    
    # Features
    df['mom_1m'] = returns.rolling(20).mean()
    df['mom_3m'] = returns.rolling(60).mean()
    df['mom_6m'] = returns.rolling(120).mean()
    df['vol_1m'] = returns.rolling(20).std()
    
    # Target (Next 5 days)
    df['target'] = returns.shift(-5).rolling(5).sum()
    
    return df.dropna()

def test_asset(prices: pd.DataFrame, ticker: str) -> dict:
    """Test ML predictive power on one asset."""
    if ticker not in prices.columns:
        return None
        
    data = create_features(prices, ticker)
    if len(data) < 250: # Need at least a year
        return None
        
    # Split
    split = int(len(data) * 0.7)
    train, test = data.iloc[:split], data.iloc[split:]
    
    X_train = train[['mom_1m', 'mom_3m', 'mom_6m', 'vol_1m']]
    y_train = train['target']
    X_test = test[['mom_1m', 'mom_3m', 'mom_6m', 'vol_1m']]
    y_test = test['target']
    
    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Metrics
    ic, _ = spearmanr(preds, y_test)
    
    # Directional Accuracy (needs significant move to count)
    # Filter out tiny moves where noise dominates
    threshold = 0.005 # 0.5% move
    mask = np.abs(y_test) > threshold
    
    if mask.sum() > 0:
        accuracy = accuracy_score(np.sign(y_test[mask]), np.sign(preds[mask]))
    else:
        accuracy = 0.5
        
    return {
        'ticker': ticker,
        'name': DISCOVERY_UNIVERSE.get(ticker, ticker),
        'ic': ic,
        'accuracy': accuracy,
        'n_samples': len(test)
    }

def run_discovery():
    print("=" * 80)
    print("   ML DISCOVERY: SEARCHING FOR COMPATIBLE ASSETS")
    print("=" * 80)
    
    prices = fetch_data(list(DISCOVERY_UNIVERSE.keys()), years=4)
    print(f"   Loaded {len(prices.columns)} assets.")
    
    results = []
    
    print("\n   Testing assets...")
    print(f"   {'Asset':<15} {'Name':<20} {'IC (Corr)':>10} {'Accuracy':>10} {'Verdict'}")
    print("   " + "-" * 75)
    
    for ticker in DISCOVERY_UNIVERSE.keys():
        res = test_asset(prices, ticker)
        if res:
            verdict = "✅ WORKS" if (res['ic'] > 0.05 and res['accuracy'] > 0.52) else "❌ FAIL"
            print(f"   {ticker:<15} {res['name']:<20} {res['ic']:>10.3f} {res['accuracy']:>10.1%}   {verdict}")
            results.append(res)
            
    # Sort by IC
    results.sort(key=lambda x: x['ic'], reverse=True)
    
    print("\n" + "=" * 80)
    print("   TOP PERFORMERS (Where the Brain Works)")
    print("=" * 80)
    
    print(f"\n   {'Rank':<5} {'Asset':<20} {'IC':>8} {'Acc':>8}")
    print("   " + "-" * 50)
    
    for i, res in enumerate(results[:15], 1):
        print(f"   {i:<5} {res['name']:<20} {res['ic']:>8.3f} {res['accuracy']:>8.1%}")
        
    return results

if __name__ == "__main__":
    run_discovery()
