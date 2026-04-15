"""
Ultimate Alpha Hunt
===================

Testing advanced alpha sources for the Cyclical/Macro Strategy.

Features to Test:
1.  Seasonality: Month of Year (Cyclicals often rally in Winter/Spring)
2.  Turn of Month: Days [-1, 0, 1, 2, 3] of month.
3.  Yield Curve Slope: 10Y - 2Y (^TNX - ^IRX?? No, need proper tickers)
    Using Proxy: IEF / SHY ratio (already tested? testing slope explicit).
4.  Volatility Regimes: Interaction term (Signal * VIX_Level).

Universe: XLB, XLI, XLE, JNK, GLD, USMV, MTUM, FXA
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

TARGET_ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'FXA', 'USMV', 'MTUM']
MACRO_ASSETS = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    tickers = list(set(TARGET_ASSETS + MACRO_ASSETS))
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

def create_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = pd.DataFrame(index=prices.index)
    
    # 1. Base Features (Momentum)
    returns = prices[ticker].pct_change()
    df['mom_1m'] = returns.rolling(20).mean()
    df['mom_3m'] = returns.rolling(60).mean()
    df['vol_1m'] = returns.rolling(20).std()
    
    # 2. Seasonality
    df['month'] = df.index.month
    df['is_winter'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
    # Turn of month (simplified: first 4 days)
    df['day'] = df.index.day
    df['turn_of_month'] = (df['day'] <= 4).astype(int)
    
    # 3. Macro (Rates) 
    # Use 10Y Yield Level
    if '^TNX' in prices.columns:
        df['rate_level'] = prices['^TNX']
    
    # 4. Yield Curve Slope (10Y approx / 2Y approx) -> IEF / SHY
    if 'IEF' in prices.columns and 'SHY' in prices.columns:
        slope = prices['IEF'] / prices['SHY']
        df['curve_slope'] = slope
        df['slope_change'] = slope.diff(20)
        
    # Target
    df['target'] = returns.shift(-5).rolling(5).sum()
    
    return df.dropna()

def run_test():
    print("=" * 80)
    print("   ULTIMATE ALPHA HUNT: Seasonality & Curve")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    
    print(f"   Testing {len(TARGET_ASSETS)} assets...")
    print("\n   Asset           Base IC    New IC      Gain      Top Feature")
    print("   " + "-" * 75)
    
    avg_gain = 0
    
    for ticker in TARGET_ASSETS:
        if ticker not in prices.columns: continue
        
        data = create_features(prices, ticker)
        if len(data) < 250: continue
        
        split = int(len(data) * 0.7)
        train, test = data.iloc[:split], data.iloc[split:]
        y_train, y_test = train['target'], test['target']
        
        # Base Model
        base_cols = ['mom_1m', 'mom_3m', 'vol_1m']
        m_base = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        m_base.fit(train[base_cols], y_train)
        ic_base = spearmanr(m_base.predict(test[base_cols]), y_test)[0]
        
        # Enhanced Model (Seasonality + Curve)
        new_cols = base_cols + ['month', 'is_winter', 'turn_of_month']
        if 'curve_slope' in data.columns:
            new_cols += ['curve_slope', 'slope_change']
        if 'rate_level' in data.columns:
            new_cols += ['rate_level']
            
        m_new = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        m_new.fit(train[new_cols], y_train)
        ic_new = spearmanr(m_new.predict(test[new_cols]), y_test)[0]
        
        gain = ic_new - ic_base
        avg_gain += gain
        
        # Top Feat
        imps = pd.Series(m_new.feature_importances_, index=new_cols).sort_values(ascending=False)
        top_feat = imps.index[0]
        if top_feat in base_cols and len(imps)>1: top_feat = imps.index[1] # Show 2nd best if base is top
        
        print(f"   {ticker:<10}      {ic_base:>6.3f}      {ic_new:>6.3f}    {gain:>6.3f}    {top_feat}")
        
    print("\n" + "=" * 80)
    print(f"   AVERAGE GAIN: {avg_gain / len(TARGET_ASSETS):.3f}")
    
if __name__ == "__main__":
    run_test()
