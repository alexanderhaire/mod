"""
ML Meta-Optimization Alpha Hunt
===============================

Using Machine Learning (Random Forest) to predict which strategy will outperform next month.
Target: Ultimate vs HRP.
Goal: Dynamic "Predictive" Allocation.

Method: Walk-Forward Validation (Train on Past, Predict Next Month).
Features: VIX, Term Structure, Relative Momentum, Volatility Regime.

RUN: python ml_meta_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA GENERATION (The Assets)
# =============================================================================

def fetch_data():
    print("🤖 Fetching Data for ML Lab...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', # Trad
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX', '^VIX3M' # Signals
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
    vix3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    # Drop signal cols from price df
    cols_to_drop = [c for c in ['^VIX', '^VIX3M'] if c in prices.columns]
    prices = prices.drop(columns=cols_to_drop)
    
    prices = prices.ffill().dropna()
    if vix is not None: vix = vix.reindex(prices.index).ffill()
    if vix3m is not None: vix3m = vix3m.reindex(prices.index).ffill()
    
    print(f"   Data: {len(prices)} days")
    return prices, vix, vix3m

def get_hrp_proxy(prices):
    # Proxy HRP (Inverse Vol) - 0.9 correlation to full HRP but faster
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

def get_ultimate(prices, vix, vix3m):
    # Ultimate Strategy logic
    vix_sig = pd.Series(0, index=prices.index)
    if vix is not None and vix3m is not None:
        ratio = (vix/vix3m).rolling(5).mean()
        vix_sig[ratio < 0.90] = 1 
        vix_sig[ratio > 1.05] = -1 
        
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

def build_features(r_ult, r_hrp, vix, vix3m):
    print("   Feature Engineering...")
    df = pd.DataFrame(index=r_ult.index)
    df['Ret_Ult'] = r_ult
    df['Ret_HRP'] = r_hrp
    
    # Target: Who wins NEXT Month?
    # We resample to Monthly for prediction
    df_m = pd.DataFrame()
    df_m['Ret_Ult'] = df['Ret_Ult'].resample('M').apply(lambda x: (1+x).prod()-1)
    df_m['Ret_HRP'] = df['Ret_HRP'].resample('M').apply(lambda x: (1+x).prod()-1)
    
    # Create Binary Target: 1 if Ult > HRP, 0 otherwise
    df_m['Target'] = (df_m['Ret_Ult'] > df_m['Ret_HRP']).astype(int)
    
    # Shift Target back by 1 (at time T, we want to predict T+1)
    # So Target at time T is the winner of T+1
    # df_m['Target'] = df_m['Target'].shift(-1) 
    # WAIT. Standard ML practice: X at T predicts Y at T+1.
    # So we want 'Next_Winner' as a column.
    df_m['Next_Winner'] = df_m['Target'].shift(-1)
    
    # Features (X) at Time T
    
    # 1. Recent Performance (Momentum)
    df_m['Ult_1m'] = df_m['Ret_Ult']
    df_m['HRP_1m'] = df_m['Ret_HRP']
    df_m['Spread_1m'] = df_m['Ult_1m'] - df_m['HRP_1m'] # Who won just now?
    
    # 2. VIX Features (Macro)
    if vix is not None:
        vix_m = vix.resample('M').last()
        df_m['VIX'] = vix_m
        df_m['VIX_Delta'] = vix_m.diff()
        
    if vix3m is not None:
        vix3m_m = vix3m.resample('M').last()
        df_m['VIX_Term'] = (vix_m / vix3m_m)
        
    # 3. Volatility Features
    # Recalculate daily vol then resample
    vol_ult = r_ult.rolling(21).std().resample('M').last()
    vol_hrp = r_hrp.rolling(21).std().resample('M').last()
    df_m['Vol_Ult'] = vol_ult
    df_m['Vol_HRP'] = vol_hrp
    df_m['Vol_Spread'] = vol_ult - vol_hrp
    
    df_m = df_m.dropna()
    return df_m

# =============================================================================
# 3. WALK-FORWARD ML
# =============================================================================

def run_ml_backtest(df):
    print(f"\n🧠 TRAINING AI MODEL (Walk-Forward)...")
    print(f"   Samples: {len(df)}")
    
    # Walk-Forward Settings
    start_train = 24 # Start after 2 years
    
    predictions = []
    probabilities = []
    
    X_cols = [c for c in df.columns if c not in ['Target', 'Next_Winner', 'Ret_Ult', 'Ret_HRP']]
    y_col = 'Next_Winner'
    
    for i in range(start_train, len(df)):
        # Train Window (Expanding)
        train = df.iloc[:i]
        test = df.iloc[i:i+1] # Predict next single month
        
        X_train = train[X_cols]
        y_train = train[y_col]
        X_test = test[X_cols]
        
        # Model
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        pred = model.predict(X_test)[0]
        prob = model.predict_proba(X_test)[0][1] # Prob of Class 1 (Ult Wins)
        
        predictions.append(pred)
        probabilities.append(prob)
        
    # Align Predictions
    # Predictions happen at time i. They apply to return at time i+1.
    # We generated 'Next_Winner' as shift(-1).
    # So prediction[0] made at index 'start_train' predicts 'Next_Winner' at 'start_train'.
    # Which corresponds to returns at date 'start_train + 1'.
    
    test_idx = df.index[start_train:]
    res_df = pd.DataFrame(index=test_idx)
    res_df['Target'] = df.loc[test_idx, 'Next_Winner']
    res_df['Pred'] = predictions
    res_df['Prob_Ult'] = probabilities
    
    # Accuracy
    acc = accuracy_score(res_df['Target'], res_df['Pred'])
    print(f"   Model Accuracy: {acc:.1%}")
    if acc > 0.55:
        print("   ✅ Predictive Edge Found!")
    else:
        print("   ❌ Model is guessing (Acc ~50%).")
        
    # Feature Importance
    importances = model.feature_importances_ # from last loop
    feat_imp = pd.Series(importances, index=X_cols).sort_values(ascending=False)
    print("\n   Top Predictors:")
    print(feat_imp.head(3))
        
    return res_df

# =============================================================================
# 4. SIMULATION
# =============================================================================

def simulate_ml_strategy(ml_res, df_data):
    # Retrieve Returns
    # Note: prediction at index T corresponds to return T+1.
    # df_data index is Monthly Date.
    # Ret_Ult at index T is the return achieved in Month T.
    # Prediction made at T-1 predicts Ret at T.
    
    # Let's align carefully.
    # ml_res index is the date prediction was made.
    # We need to apply this prediction to the NEXT month's return.
    
    # Shift predictions forward by 1 month to match returns
    # ml_res['Pred'] is prediction for the 'Next_Winner'.
    # 'Next_Winner' at index T is based on Ret at T+1.
    # So Pred at T should be matched with Ret at T+1.
    
    sim = pd.DataFrame(index=ml_res.index)
    sim['Ret_Ult'] = df_data['Ret_Ult'].shift(-1).loc[ml_res.index] # Future return
    sim['Ret_HRP'] = df_data['Ret_HRP'].shift(-1).loc[ml_res.index]
    sim['Prob_Ult'] = ml_res['Prob_Ult']
    
    sim = sim.dropna()
    
    # Strategy Logic
    # 1. 50/50 Benchmark
    sim['Ret_5050'] = 0.5 * sim['Ret_Ult'] + 0.5 * sim['Ret_HRP']
    
    # 2. ML Binary (0 or 100)
    # If Prob > 0.5 -> 100% Ult, else 100% HRP
    sim['W_Ult_Bin'] = (sim['Prob_Ult'] > 0.5).astype(float)
    sim['Ret_ML_Bin'] = sim['W_Ult_Bin'] * sim['Ret_Ult'] + (1 - sim['W_Ult_Bin']) * sim['Ret_HRP']
    
    # 3. ML Confidence (Threshold)
    # If Prob > 0.60 -> 100% Ult
    # If Prob < 0.40 -> 100% HRP
    # Else -> 50/50
    conditions = [
        (sim['Prob_Ult'] > 0.60),
        (sim['Prob_Ult'] < 0.40)
    ]
    choices = [1.0, 0.0]
    sim['W_Ult_Conf'] = np.select(conditions, choices, default=0.5)
    sim['Ret_ML_Conf'] = sim['W_Ult_Conf'] * sim['Ret_Ult'] + (1 - sim['W_Ult_Conf']) * sim['Ret_HRP']
    
    def get_stats(r, name):
        # Monthly returns to annualized
        ann = r.mean() * 12
        vol = r.std() * np.sqrt(12)
        sharpe = ann / vol if vol > 0 else 0
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f}")
        return sharpe
        
    print("\n   STRATEGY PERFORMANCE (Out-of-Sample):")
    print("-" * 75)
    s_base = get_stats(sim['Ret_5050'], "Static 50/50")
    s_ml = get_stats(sim['Ret_ML_Conf'], "ML Confidence Switch")
    
    if s_ml > s_base:
        print(f"\n   ✅ ML WINS: Improvement of +{s_ml - s_base:.2f} Sharpe.")
    else:
        print(f"\n   ❌ ML FAILS: Advanced math couldn't beat simple average.")

if __name__ == "__main__":
    prices, vix, vix3m = fetch_data()
    r_ult = get_ultimate(prices, vix, vix3m)
    r_hrp = get_hrp_proxy(prices)
    
    df_features = build_features(r_ult, r_hrp, vix, vix3m)
    res_ml = run_ml_backtest(df_features)
    simulate_ml_strategy(res_ml, df_features)
