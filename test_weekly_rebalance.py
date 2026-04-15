"""
Weekly vs Daily Rebalancing Test
================================

Testing if slowing down the strategy improves performance.
Baseline: Daily Rebalance (Signals generated daily for 5-day horizon).
Challenger: Weekly Rebalance (Signals generated only on Friday for next week).

Theory:
Daily is more reactive to Macro shocks.
Weekly has lower noise/turnover.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'EFA'] # Top performers from Deep Discovery
MACRO = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    tickers = list(set(ASSETS + MACRO))
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

def create_features(prices: pd.DataFrame) -> tuple:
    df = pd.DataFrame(index=prices.index)
    
    # Simple Features for speed
    if '^TNX' in prices.columns: df['rate_chg'] = prices['^TNX'].diff(20)
    
    asset_map = {}
    for t in ASSETS:
        r = prices[t].pct_change()
        df[f'{t}_mom'] = r.rolling(20).mean()
        asset_map[t] = [f'{t}_mom', 'rate_chg'] if 'rate_chg' in df.columns else [f'{t}_mom']
        
    targets = prices.pct_change().shift(-1) # 1 day forward return
    return df, targets, asset_map

def run_test():
    print("=" * 80)
    print("   FREQUENCY TEST: DAILY vs WEEKLY")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    features, targets, asset_map = create_features(prices)
    
    # Train/Test Split
    split = int(len(prices) * 0.7)
    train_end = prices.index[split]
    
    # Train Models
    models = {}
    train_idx = prices.index[:split]
    for t in ASSETS:
        X = features.loc[train_idx][asset_map[t]].dropna()
        y = targets.loc[X.index][t]
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        m.fit(X, y)
        models[t] = m
        
    # Predict Full OOS
    oos_idx = prices.index[split:]
    signals = pd.DataFrame(index=oos_idx, columns=ASSETS)
    
    for i in range(len(oos_idx)):
        idx = oos_idx[i]
        for t in ASSETS:
            row = features.loc[idx:idx][asset_map[t]]
            if row.isnull().any().any():
                signals.iloc[i][t] = 0
            else:
                signals.iloc[i][t] = models[t].predict(row)[0]
                
    signals = (signals > 0).astype(int) # Binary Long Only
    
    # 1. Daily Strategy
    # Rebalance every day based on signal
    daily_w = signals.div(signals.sum(axis=1), axis=0).fillna(0)
    daily_ret = (daily_w * targets.loc[oos_idx]).sum(axis=1)
    
    # 2. Weekly Strategy
    # Rebalance only on Fridays (weekday=4)
    # Hold constant weights Mon-Thu
    weekly_w = pd.DataFrame(0, index=oos_idx, columns=ASSETS)
    current_w = pd.Series(0, index=ASSETS)
    
    for dt in oos_idx:
        if dt.weekday() == 4: # Friday
            # Update weights based on FRIDAY'S signal
            if dt in daily_w.index:
                current_w = daily_w.loc[dt]
        weekly_w.loc[dt] = current_w
        
    weekly_ret = (weekly_w * targets.loc[oos_idx]).sum(axis=1)
    
    # Stats
    def report(r, name):
        ann = r.mean() * 252
        sharpe = r.mean() / r.std() * np.sqrt(252)
        print(f"   {name:<10} Sharpe: {sharpe:.2f}  AnnRet: {ann:.1%}")
        
    report(daily_ret, "Daily")
    report(weekly_ret, "Weekly")
    
    if daily_ret.mean() > weekly_ret.mean():
        print("\n   ✅ CONCLUSION: Daily is better (Faster reaction).")
    else:
        print("\n   ✅ CONCLUSION: Weekly is better (Less noise).")

if __name__ == "__main__":
    run_test()
