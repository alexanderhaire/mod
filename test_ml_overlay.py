
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def run_ml_overlay_test():
    print("=" * 80)
    print("   AI META-OPTIMIZATION: Can ML predict when Golden Omni fails?")
    print("=" * 80)
    
    # 1. Fetch Data
    tickers = [
        'SPY', 'TLT', 'GLD', 'XLE', 'BTC-USD', 'ETH-USD', '^VIX'
    ]
    print(f"Fetching data: {tickers}...")
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna(subset=['SPY'])
    rets = prices.pct_change().fillna(0)
    
    # 2. Reconstruct Golden Omni Equity Curve (Simplified for Speed)
    # -----------------------------------------------------------
    # Crypto Leg
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True)
    
    r_crypto = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in rets.columns:
        r_crypto = rets['BTC-USD'].fillna(0)
        # Apply safety filter
        r_crypto[~is_crypto_safe] = 0.0 # To cash
        
    missing_crypto = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in prices.columns:
        btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
        missing_crypto[~btc_avail] = 0.40 * rets['SPY'][~btc_avail]

    # Regime Logic
    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    is_bull = (spy_p > ma200).shift(1).fillna(False)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation = ((xle_p > ma200_xle) & (~is_bull)).shift(1).fillna(False)

    # Allocations
    r_bull = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.05 * rets.get('GLD', 0) + 0.40 * r_crypto + missing_crypto)
    r_bear = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto + missing_crypto)
    r_inf = (0.15 * rets['SPY'] + 0.35 * rets.get('XLE', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto + missing_crypto)
    
    r_omni = pd.Series(0.0, index=rets.index)
    r_omni[is_bull] = r_bull[is_bull]
    r_omni[(~is_bull) & (~is_inflation)] = r_bear[(~is_bull) & (~is_inflation)]
    r_omni[(~is_bull) & (is_inflation)] = r_inf[(~is_bull) & (is_inflation)]
    
    omni_equity = (1 + r_omni).cumprod()
    
    # 3. Feature Engineering (The "AI" Part)
    # -------------------------------------
    print("\nTraining Meta-Labeling Model (Random Forest)...")
    
    features = pd.DataFrame(index=prices.index)
    
    # Volatility
    features['spy_vol_21'] = rets['SPY'].rolling(21).std()
    features['tlt_vol_21'] = rets['TLT'].rolling(21).std()
    features['vix'] = prices.get('^VIX', pd.Series(20, index=prices.index))
    
    # Momentum / Trend
    features['spy_mom_21'] = prices['SPY'].pct_change(21)
    features['tlt_mom_21'] = prices['TLT'].pct_change(21)
    features['gld_mom_21'] = prices['GLD'].pct_change(21)
    
    # Correlations
    features['corr_spy_tlt'] = rets['SPY'].rolling(63).corr(rets['TLT'])
    features['corr_spy_gld'] = rets['SPY'].rolling(63).corr(rets['GLD'])
    
    # Target: Will Omni lose money next week? (5 days)
    # If Next Week Return < -1.0% -> Label 1 (Danger), Else 0
    omni_fwd = r_omni.rolling(5).sum().shift(-5)
    target = (omni_fwd < -0.01).astype(int)
    
    # Clean
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    data_df = pd.concat([features, target.rename('target')], axis=1).dropna()
    
    if len(data_df) < 500:
        print("Not enough data for ML.")
        return

    # 4. Walk-Forward Simulation
    # --------------------------
    X = data_df[features.columns]
    y = data_df['target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    # Train on rolling window, predict next month
    predictions = pd.Series(0, index=X.index)
    
    start_idx = 500 # 2 years warmup
    step = 21 # Monthly Re-train
    
    print(f"   Starting walk-forward validation (Train 2y -> Predict 1mo)...")
    
    for i in range(start_idx, len(X), step):
        train_X = X.iloc[:i]
        train_y = y.iloc[:i]
        
        # Only train if we have both classes
        if train_y.nunique() > 1:
            model.fit(train_X, train_y)
            
            end_idx = min(i+step, len(X))
            test_X = X.iloc[i:end_idx]
            
            pred_probs = model.predict_proba(test_X)[:, 1] # Prob of Danger
            predictions.iloc[i:end_idx] = pred_probs
            
    # 5. Apply " AI Filter"
    # ---------------------
    # If Prob(Danger) > Threshold -> Go to Cash (or 50% Exp)
    threshold = 0.65
    safety_signal = (predictions < threshold).astype(float) # 1 = Safe, 0 = Danger
    
    # Shift signal to apply to NEXT period returns
    # Predictions aligns with X features (today). We use it to trade tomorrow.
    trading_signal = safety_signal.shift(1).fillna(1)
    
    # Hybrid Strategy: Omni * Signal
    r_hybrid = r_omni.loc[predictions.index] * trading_signal
    
    # 6. Compare
    # ----------
    r_base = r_omni.loc[predictions.index]
    
    # Stats
    def get_stats(r):
        py = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = py / vol if vol > 0 else 0
        dd = ((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min()
        return sharpe, py, dd
        
    s_base, r_base_ann, dd_base = get_stats(r_base)
    s_hyb, r_hyb_ann, dd_hyb = get_stats(r_hybrid)
    
    print("\n" + "="*60)
    print("   RESULTS: Base Omni vs AI-Filtered Omni")
    print("="*60)
    print(f"{'Metric':<20} | {'Base Omni':<15} | {'AI Hybrid':<15}")
    print("-" * 60)
    print(f"{'Sharpe Ratio':<20} | {s_base:<15.2f} | {s_hyb:<15.2f}")
    print(f"{'Annual Return':<20} | {r_base_ann:<15.1%} | {r_hyb_ann:<15.1%}")
    print(f"{'Max Drawdown':<20} | {dd_base:<15.1%} | {dd_hyb:<15.1%}")
    
    # Signal Analysis
    n_danger = (trading_signal == 0).sum()
    pct_cash = n_danger / len(trading_signal)
    print(f"\n   AI chose to stay in CASH for {pct_cash:.1%} of time.")
    
    if s_hyb > s_base * 1.05:
        print("\n   ✅ SUCCESS: ML Filter significantly improved Sharpe.")
    else:
        print("\n   ❌ FAILURE: ML Filter did not add value (or over-filtered).")

if __name__ == "__main__":
    run_ml_overlay_test()
