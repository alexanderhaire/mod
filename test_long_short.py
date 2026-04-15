"""
Long/Short Strategy Test
========================

Exploiting both ends of the prediction spectrum.
Long the High IC Assets: XLB, XLE, MTUM, EFA, GLD
Short the Negative IC Assets (Inverse Signal): XLU (Utilities), SLV (Silver)?
Or just Short the lowest predicted return?

Strategy:
1.  Rank assets by Predicted Return.
2.  Long Top 3.
3.  Short Bottom 3.
4.  Compare vs Long-Only.

Hypothesis: Shorting provides downside protection and alpha during regime shifts.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Expanded Universe (Long candidates + Short candidates)
# Include Negative IC assets to see if model dislikes them correctly
ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'EFA', 'MTUM', 'qual', 'XLU', 'SLV'] 
MACRO = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    tickers = list(set([a.upper() for a in ASSETS] + MACRO))
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

def create_features(prices: pd.DataFrame) -> tuple:
    df = pd.DataFrame(index=prices.index)
    feat_cols = []
    
    # Macro
    if '^TNX' in prices.columns:
        df['rate_change'] = prices['^TNX'].diff(20)
        feat_cols += ['rate_change']
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        df['credit_spread'] = prices['JNK'] / prices['IEF']
        feat_cols += ['credit_spread']
        
    asset_map = {}
    for t in prices.columns:
        if t not in [a.upper() for a in ASSETS]: continue
        r = prices[t].pct_change()
        cols = []
        for w in [20, 60]:
            c = f'{t}_mom_{w}'
            df[c] = r.rolling(w).mean()
            cols.append(c)
        c = f'{t}_vol_20'
        df[c] = r.rolling(20).std()
        cols.append(c)
        asset_map[t] = cols + feat_cols
        
    targets = prices.pct_change().shift(-1)
    return df, targets, asset_map

def run_backtest():
    print("=" * 80)
    print("   LONG/SHORT STRATEGY TEST")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    features, targets, asset_map = create_features(prices)
    common = features.dropna().index.intersection(targets.dropna().index)
    
    # Split
    split_idx = int(len(common) * 0.7)
    train_end = common[split_idx]
    
    # Train Models (IS)
    models = {}
    train_idx = common[:split_idx]
    
    for t in asset_map.keys():
        cols = asset_map[t]
        X = features.loc[train_idx][cols]
        y = targets.loc[train_idx][t]
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        m.fit(X, y)
        models[t] = m
        
    # Predict (OOS)
    oos_idx = common[split_idx:]
    preds = pd.DataFrame(index=oos_idx, columns=asset_map.keys())
    
    for i in range(len(oos_idx)):
        idx = oos_idx[i]
        for t in asset_map.keys():
            row = features.loc[idx:idx][asset_map[t]] # keep DF
            preds.loc[idx, t] = models[t].predict(row)[0]
            
    # Strategy Logic
    # 1. Long Only (Top 3)
    # 2. Long/Short (Top 3 - Bottom 3)
    
    long_ret = []
    ls_ret = []
    
    aligned_rets = targets.loc[oos_idx]
    
    for i in range(len(preds)):
        row = preds.iloc[i].sort_values(ascending=False)
        top3 = row.index[:3]
        bot3 = row.index[-3:]
        
        # Returns for next day
        day_rets = aligned_rets.iloc[i]
        
        # Long Only
        l_r = day_rets[top3].mean()
        long_ret.append(l_r)
        
        # Long/Short (100% Long, 100% Short -> Net 0? No, usually 130/30 or 50/50)
        # Let's do Dollar Neutral: Long Top 3, Short Bottom 3.
        s_r = day_rets[bot3].mean()
        ls_r = l_r - s_r # Short return is -(return)
        ls_ret.append(ls_r)
        
    long_ret = pd.Series(long_ret, index=oos_idx)
    ls_ret = pd.Series(ls_ret, index=oos_idx)
    
    # Metrics
    def stats(r, name):
        ann_ret = r.mean() * 252
        sharpe = r.mean() / r.std() * np.sqrt(252)
        dd = (1+r).cumprod() / (1+r).cumprod().cummax() - 1
        print(f"   {name:<15} Ret: {ann_ret:.1%}  Sharpe: {sharpe:.2f}  MaxDD: {dd.min():.1%}")

    print("\n   === RESULTS (OOS) ===")
    stats(long_ret, "Long Only (Top3)")
    stats(ls_ret, "Long/Short (Neu)")
    
    # Pass check
    if ls_ret.mean() / ls_ret.std() * np.sqrt(252) > long_ret.mean() / long_ret.std() * np.sqrt(252):
        print("\n   ✅ CONCLUSION: Shorting adds value (Alpha).")
    else:
        print("\n   ❌ CONCLUSION: Shorting degrades quality (Hard to borrow/Costly).")

if __name__ == "__main__":
    run_backtest()
