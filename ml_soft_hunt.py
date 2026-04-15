"""
Soft-ML Allocation Alpha Hunt
=============================

Refining the ML approach.
Shift from Binary Switching (100/0) to Proportional Tilting (e.g. 60/40).
Model: Random Forest Regressor (Predicting the *Spread* between Ultimate and HRP).
Features: Macro (VIX, Yield Curve Proxy) + Momentum.

Hypothesis: Small tilts based on high confidence capture alpha without the risk of binary ruin.

RUN: python ml_soft_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def fetch_data():
    print("🧠 Fetching Data for Soft-ML Lab...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'SHY', 'QQQ', # Trad
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX', '^TNX' # Macro: VIX, 10Y Yield
    ]
    data = yf.download(tickers, start='2015-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    tnx = prices['^TNX'].copy() if '^TNX' in prices.columns else None
    
    # Drop signal cols
    cols_to_drop = [c for c in ['^VIX', '^TNX'] if c in prices.columns]
    prices = prices.drop(columns=cols_to_drop)
    
    prices = prices.ffill().dropna()
    if vix is not None: vix = vix.reindex(prices.index).ffill()
    if tnx is not None: tnx = tnx.reindex(prices.index).ffill()
    
    print(f"   Data: {len(prices)} days")
    return prices, vix, tnx

def get_hrp_proxy(prices):
    # Proxy HRP (Inverse Vol)
    returns = prices.pct_change().dropna()
    weights = pd.DataFrame(index=returns.index, columns=prices.columns)
    lookback = 126
    m_dates = returns.resample('M').last().index
    for t in m_dates:
        hist = returns[returns.index <= t].tail(lookback)
        if len(hist) < 60: continue
        vol = hist.std()
        w = (1/vol) / (1/vol).sum()
        try: weights.loc[t] = w
        except: pass
    weights = weights.ffill().dropna()
    return (weights.shift(1) * returns).sum(axis=1)

def get_ultimate(prices, vix):
    # Ultimate Strategy logic (Simplified to VIX term structure proxy via SMA)
    # We don't have VIX3M here easily, so use VIX vs SMA(VIX)
    # VIX > SMA(20) = Fear (Risk Off). VIX < SMA(20) = Calm (Risk On).
    
    vix_ma = vix.rolling(20).mean()
    vix_sig = pd.Series(0, index=prices.index)
    vix_sig[vix < vix_ma] = 1 # Bullish
    vix_sig[vix > vix_ma] = -1 # Bearish
        
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(1, len(prices)):
        vs = vix_sig.iloc[i]
        # Trad
        if vs > 0: w_t = {'SPY': 0.45, 'TLT': 0.10}
        elif vs < 0: w_t = {'SPY': 0.15, 'TLT': 0.35}
        else: w_t = {'SPY': 0.30, 'TLT': 0.22}
        
        # Crypto
        if 'BTC-USD' in prices.columns:
            btc = prices['BTC-USD'].iloc[i]
            prev = prices['BTC-USD'].iloc[i-1] if i > 0 else btc
            w_c = {'BTC-USD': 0.25} if btc > prev else {'BTC-USD': 0.10}
            
        for t, w in w_t.items(): 
             if t in prices.columns: weights.iloc[i][t] = w
        for t, w in w_c.items():
             if t in prices.columns: weights.iloc[i][t] = w
    return (weights.shift(1) * prices.pct_change()).sum(axis=1)

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def build_features(r_ult, r_hrp, vix, tnx):
    print("   Feature Engineering (Macro + Relative)...")
    df = pd.DataFrame(index=r_ult.index)
    df['Ret_Ult'] = r_ult
    df['Ret_HRP'] = r_hrp
    
    # Monthly Resample
    df_m = pd.DataFrame()
    df_m['Ret_Ult'] = df['Ret_Ult'].resample('M').apply(lambda x: (1+x).prod()-1)
    df_m['Ret_HRP'] = df['Ret_HRP'].resample('M').apply(lambda x: (1+x).prod()-1)
    
    # Target: Spread (Ult - HRP) Next Month
    df_m['Spread'] = df_m['Ret_Ult'] - df_m['Ret_HRP']
    df_m['Next_Spread'] = df_m['Spread'].shift(-1)
    
    # Features
    
    # 1. Momentum Spread
    df_m['Ult_3m'] = df_m['Ret_Ult'].rolling(3).mean()
    df_m['HRP_3m'] = df_m['Ret_HRP'].rolling(3).mean()
    df_m['Mom_Spread'] = df_m['Ult_3m'] - df_m['HRP_3m']
    
    # 2. Macro (Yield Curve & VIX)
    if vix is not None:
        df_m['VIX'] = vix.resample('M').last()
        df_m['VIX_Delta'] = df_m['VIX'].diff()
        
    if tnx is not None:
        df_m['TNX'] = tnx.resample('M').last() # 10 Year Yield
        # Crude yield curve slope proxy: TNX trend? 
        # Or better: if TNX is rising, bonds (HRP) suffer.
        df_m['TNX_Delta'] = df_m['TNX'].diff()
        
    # 3. Volatility
    vol_ult = r_ult.rolling(63).std().resample('M').last()
    vol_hrp = r_hrp.rolling(63).std().resample('M').last()
    df_m['Vol_Spread'] = vol_ult - vol_hrp
    
    df_m = df_m.dropna()
    return df_m

# =============================================================================
# 3. WALK-FORWARD REGRESSION
# =============================================================================

def run_ml_regressor(df):
    print(f"\n🧠 TRAINING REGRESSOR (Predicting Spread)...")
    
    start_train = 36 # 3 years
    
    predictions = []
    
    X_cols = [c for c in df.columns if c not in ['Next_Spread', 'Ret_Ult', 'Ret_HRP']]
    y_col = 'Next_Spread'
    
    for i in range(start_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i:i+1]
        
        X_train = train[X_cols]
        y_train = train[y_col]
        X_test = test[X_cols]
        
        # Regressor
        model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        
    idx = df.index[start_train:]
    res_df = pd.DataFrame(index=idx)
    res_df['Actual_Spread'] = df.loc[idx, 'Next_Spread']
    res_df['Pred_Spread'] = predictions
    
    # Correlation
    corr = res_df['Actual_Spread'].corr(res_df['Pred_Spread'])
    r2 = r2_score(res_df['Actual_Spread'], res_df['Pred_Spread'])
    
    print(f"   Model IC (Correlation): {corr:.3f}")
    print(f"   Model R2: {r2:.3f}")
    
    # Feature Importance
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X_cols).sort_values(ascending=False)
    print("\n   Top Predictors:")
    print(feat_imp.head(3))
    
    return res_df

# =============================================================================
# 4. STRATEGY SIMULATION (TILT)
# =============================================================================

def simulate_tilt(ml_res, df_data):
    # Align
    # ml_res['Pred_Spread'] at time T predicts Spread T+1.
    # We want to use this to weight T+1 returns.
    
    sim = pd.DataFrame(index=ml_res.index)
    sim['Ret_Ult'] = df_data['Ret_Ult'].shift(-1).loc[ml_res.index]
    sim['Ret_HRP'] = df_data['Ret_HRP'].shift(-1).loc[ml_res.index]
    sim['Pred_Spread'] = ml_res['Pred_Spread']
    sim = sim.dropna()
    
    # Strategy Logic: Tilt
    # Base Weight = 0.50
    # Sensitivity: How much to tilt?
    # If Pred Spread is +0.02 (2%), maybe tilt 20%?
    # W_Ult = 0.50 + (Pred_Spread * Multiplier)
    # Clip between 0.20 and 0.80 for safety (Soft ML)
    
    multiplier = 5.0 # If spread is 2%, add 10% to weight
    
    sim['Tilt'] = sim['Pred_Spread'] * multiplier
    sim['W_Ult'] = 0.50 + sim['Tilt']
    sim['W_Ult'] = sim['W_Ult'].clip(0.20, 0.80) # Safety bands
    sim['W_HRP'] = 1.0 - sim['W_Ult']
    
    sim['Ret_ML_Tilt'] = sim['W_Ult'] * sim['Ret_Ult'] + sim['W_HRP'] * sim['Ret_HRP']
    sim['Ret_5050'] = 0.5 * sim['Ret_Ult'] + 0.5 * sim['Ret_HRP']
    
    def get_stats(r, name):
        ann = r.mean() * 12
        vol = r.std() * np.sqrt(12)
        sharpe = ann / vol if vol > 0 else 0
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f}")
        return sharpe
        
    print("\n   SOFT-ML STRATEGY PERFORMANCE:")
    print("-" * 75)
    s_base = get_stats(sim['Ret_5050'], "Static 50/50")
    s_tilt = get_stats(sim['Ret_ML_Tilt'], "ML Tilt (Prop. Alloc)")
    
    print("-" * 75)
    if s_tilt > s_base + 0.05:
         print(f"✅ SOFT ML WINS: Tilt improved Sharpe to {s_tilt:.2f}")
    elif s_tilt > s_base:
         print(f"⚠️ MARGINAL: Tiny improvement ({s_tilt:.2f})")
    else:
         print(f"❌ ML FAILS: Tilting added noise. Stick to 50/50.")

if __name__ == "__main__":
    prices, vix, tnx = fetch_data()
    r_ult = get_ultimate(prices, vix)
    r_hrp = get_hrp_proxy(prices)
    
    df_features = build_features(r_ult, r_hrp, vix, tnx)
    res_ml = run_ml_regressor(df_features)
    simulate_tilt(res_ml, df_features)
