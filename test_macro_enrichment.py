"""
Macro Factor Enrichment Test
============================

Testing if adding Macroeconomic Features improves predictive power.
Hypothesis: Cyclical & Credit assets are driven by Rates and Dollar.

Macro Features:
1. US 10Y Yield (^TNX)
2. US Dollar (UUP)
3. Credit Spread (JNK / IEF ratio)
4. Yield Curve Proxy (IEF / SHY ratio)

Target Assets:
XLB, XLI, XLE, JNK, GLD
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UNIVERSE
# =============================================================================

TARGET_ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'FXA', 'USMV', 'MTUM']
MACRO_ASSETS = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK'] # JNK used for spread

# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(years: int = 5) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    tickers = list(set(TARGET_ASSETS + MACRO_ASSETS))
    print(f"Fetching data for {len(tickers)} assets + macro...")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    prices = prices.dropna(how='all').ffill().dropna()
    return prices

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_rate_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create macro features."""
    df = pd.DataFrame(index=prices.index)
    
    # 1. 10Y Yield (^TNX)
    if '^TNX' in prices.columns:
        tnx = prices['^TNX']
        df['rate_level'] = tnx
        df['rate_change_1m'] = tnx.diff(20)
        df['rate_trend'] = tnx - tnx.rolling(60).mean()
        
    # 2. Dollar (UUP)
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        df['dollar_trend'] = uup.pct_change().rolling(60).mean()
        df['dollar_vol'] = uup.pct_change().rolling(20).std()
        
    # 3. Yield Curve Proxy (IEF / SHY) ~ (7-10y / 1-3y)
    # Higher = flatter curve (usually), Lower = steeper? 
    # Actually: IEF (Longer duration) drops more when rates rise.
    # Yield Curve Inversion usually means Short Rates > Long Rates.
    # Price Ratio: IEF/SHY.
    if 'IEF' in prices.columns and 'SHY' in prices.columns:
        curve = prices['IEF'] / prices['SHY']
        df['curve_level'] = curve
        df['curve_change'] = curve.diff(20)
        
    # 4. Credit Spreads (JNK / IEF)
    # Risk On = JNK outperforms IEF.
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        credit = prices['JNK'] / prices['IEF']
        df['credit_spread'] = credit
        df['credit_trend'] = credit.pct_change().rolling(20).mean()
        
    return df.ffill().dropna()

def create_asset_features(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standard momentum features."""
    returns = prices[ticker].pct_change()
    df = pd.DataFrame(index=prices.index)
    
    df['mom_1m'] = returns.rolling(20).mean()
    df['mom_3m'] = returns.rolling(60).mean()
    df['vol_1m'] = returns.rolling(20).std()
    
    # Target
    df['target'] = returns.shift(-5).rolling(5).sum()
    
    return df.dropna()

# =============================================================================
# TEST
# =============================================================================

def run_test():
    print("=" * 80)
    print("   MACRO FACTOR ENRICHMENT TEST")
    print("   Does knowing Rates & Dollar help predict Cyclicals?")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    macro_feats = create_rate_features(prices)
    
    print(f"   Generated {len(macro_feats.columns)} Macro Features.")
    print("   Testing Base vs Macro-Enriched Models...")
    print("\n   Asset           Base IC    Macro IC    Gain      Verdict")
    print("   " + "-" * 70)
    
    results = []
    
    for ticker in TARGET_ASSETS:
        if ticker not in prices.columns: continue
        
        asset_feats = create_asset_features(prices, ticker)
        
        # Merge
        full_data = pd.concat([asset_feats, macro_feats], axis=1).dropna()
        if len(full_data) < 250: continue
        
        # Split
        split = int(len(full_data) * 0.7)
        train, test = full_data.iloc[:split], full_data.iloc[split:]
        
        y_train = train['target']
        y_test = test['target']
        
        # 1. Base Model (Mom only)
        base_cols = ['mom_1m', 'mom_3m', 'vol_1m']
        m_base = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        m_base.fit(train[base_cols], y_train)
        preds_base = m_base.predict(test[base_cols])
        ic_base, _ = spearmanr(preds_base, y_test)
        
        # 2. Macro Model (Mom + Macro)
        macro_cols = base_cols + list(macro_feats.columns)
        # Ensure cols exist
        macro_cols = [c for c in macro_cols if c in full_data.columns]
        
        m_macro = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        m_macro.fit(train[macro_cols], y_train)
        preds_macro = m_macro.predict(test[macro_cols])
        ic_macro, _ = spearmanr(preds_macro, y_test)
        
        # Result
        gain = ic_macro - ic_base
        verdict = "✅ IMPROVED" if gain > 0.02 else ("❌ NO HELP" if gain < 0 else "➖ NEUTRAL")
        
        print(f"   {ticker:<10}      {ic_base:>6.3f}      {ic_macro:>6.3f}    {gain:>6.3f}    {verdict}")
        
        # Check Feature Importance for Macro
        if gain > 0:
            imps = pd.Series(m_macro.feature_importances_, index=macro_cols).sort_values(ascending=False)
            top_macro = [c for c in imps.index if c in macro_feats.columns][:2]
            if top_macro:
                print(f"                                      Top Macro: {', '.join(top_macro)}")
        
        results.append({'asset': ticker, 'ic_base': ic_base, 'ic_macro': ic_macro, 'gain': gain})
        
    # Summary
    avg_gain = np.mean([r['gain'] for r in results])
    print("\n" + "=" * 80)
    print(f"   AVERAGE IC GAIN: {avg_gain:.3f}")
    
    if avg_gain > 0.02:
        print("   ✅ SUCCESS: Macro factors significantly improve prediction.")
    else:
        print("   ❌ FAILURE: Macro factors add noise or no value (already priced in momentum?).")

if __name__ == "__main__":
    run_test()
