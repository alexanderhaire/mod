
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def calculate_hurst(series, lags=[2, 20]):
    """
    Calculate Hurst Exponent to test for Trend (H > 0.5) or Mean Reversion (H < 0.5).
    """
    # Simple Rescaled Range or Variance Ratio test proxy
    # We'll use the variance of diffs method
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0 

def rolling_hurst(series, window=63):
    return series.rolling(window).apply(lambda x: calculate_hurst(x), raw=True)

def run_regression_shootout():
    print("=" * 80)
    print("   MATH SHOOTOUT: Can Advanced Regression Beat the Golden Rule?")
    print("   Models: Logistic, Ridge, XGBoost, SVM")
    print("   Features: Hurst Exponent (Chaos Theory), Volatility, Momentum")
    print("=" * 80)
    
    # 1. Fetch Data
    tickers = ['SPY', 'TLT', 'GLD', 'XLE', 'BTC-USD', 'ETH-USD', '^VIX']
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
    
    # 2. Reconstruct Golden Omni (The Benchmark)
    # ------------------------------------------
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True)
    
    r_crypto = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in rets.columns:
        r_crypto = rets['BTC-USD'].fillna(0)
        r_crypto[~is_crypto_safe] = 0.0
    
    missing_crypto = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in prices.columns:
        btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
        missing_crypto[~btc_avail] = 0.40 * rets['SPY'][~btc_avail]

    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    is_bull = (spy_p > ma200).shift(1).fillna(False)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation = ((xle_p > ma200_xle) & (~is_bull)).shift(1).fillna(False)

    r_bull = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.05 * rets.get('GLD', 0) + 0.40 * r_crypto + missing_crypto)
    r_bear = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto + missing_crypto)
    r_inf = (0.15 * rets['SPY'] + 0.35 * rets.get('XLE', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto + missing_crypto)
    
    r_omni = pd.Series(0.0, index=rets.index)
    r_omni[is_bull] = r_bull[is_bull]
    r_omni[(~is_bull) & (~is_inflation)] = r_bear[(~is_bull) & (~is_inflation)]
    r_omni[(~is_bull) & (is_inflation)] = r_inf[(~is_bull) & (is_inflation)]
    
    # 3. Feature Engineering ("The Math")
    # -----------------------------------
    print("\nCalculating Advanced 'Math' Features (Hurst, etc)...")
    
    features = pd.DataFrame(index=prices.index)
    
    # Hurst Exponent (Chaos/Fractal) - Is SPY Trending or Mean Reverting?
    # Uses a simplified rolling approx
    features['spy_hurst'] = rolling_hurst(prices['SPY'].apply(np.log), window=126).fillna(0.5)
    
    # Rolling Volatility Ratios
    v21 = rets['SPY'].rolling(21).std()
    v63 = rets['SPY'].rolling(63).std()
    features['vol_ratio'] = (v21 / v63).fillna(1.0)
    
    # Correlation Regimes
    features['corr_stock_bond'] = rets['SPY'].rolling(63).corr(rets['TLT']).fillna(0)
    
    # Momentum Z-Scores
    mom21 = prices['SPY'].pct_change(21)
    features['mom_z'] = ((mom21 - mom21.rolling(252).mean()) / mom21.rolling(252).std()).fillna(0)
    
    # VIX Curve (Contango/Backwardation Proxy)
    # We only have spot VIX, so use VIX vs MA
    vix = prices.get('^VIX', pd.Series(20, index=prices.index))
    features['vix_trend'] = (vix / vix.rolling(20).mean()).fillna(1.0)
    
    # Target: Omni Forward Return (5 days)
    # Regression Target: Actual Return
    # Classification Target: 1 if Return > 0
    fwd_ret = r_omni.rolling(5).sum().shift(-5)
    y_reg = fwd_ret.fillna(0)
    y_clf = (fwd_ret > 0).astype(int)
    
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 4. Define Models
    # ----------------
    models = {
        'Logistic Class (Lin)': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())]),
        'SVM Class (NonLin)': Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf', probability=True))]),
        'Ridge Reg (Linear)': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
        'Boost Reg (Tree)': GradientBoostingRegressor(n_estimators=50, max_depth=3)
    }
    
    results = {}
    
    # 5. Walk-Forward Simulation
    # --------------------------
    start_idx = 756 # 3 years warmup
    step = 21 # Monthly
    
    print(f"Starting Walk-Forward Validation ({len(prices)} days)...")
    
    for name, model in models.items():
        print(f"   Testing {name}...", end=" ", flush=True)
        
        preds = pd.Series(0.0, index=features.index)
        
        for i in range(start_idx, len(features), step):
            train_X = features.iloc[:i]
            
            # Use Reg or Clf target
            if 'Reg' in name or 'Boost' in name:
                train_y = y_reg.iloc[:i]
            else:
                train_y = y_clf.iloc[:i]
                
            # Fit
            try:
                if 'Reg' in name or 'Boost' in name:
                    model.fit(train_X, train_y)
                    # Predict Next Month
                    end = min(i+step, len(features))
                    test_X = features.iloc[i:end]
                    p = model.predict(test_X)
                    # Convert return forecast to signal (Long if > 0)
                    preds.iloc[i:end] = (p > 0).astype(float)
                else:
                    if train_y.nunique() > 1:
                        model.fit(train_X, train_y)
                        end = min(i+step, len(features))
                        test_X = features.iloc[i:end]
                        p = model.predict(test_X)
                        preds.iloc[i:end] = p
            except:
                pass
                
        # Calculate Strategy Returns
        # Signal shifted by 1 day to trade
        sig = preds.shift(1).fillna(1)
        
        # Strategy: Go Long Omni if Signal=1, else Cash
        r_strat = r_omni * sig
        
        sharpe = r_strat.mean() / r_strat.std() * np.sqrt(252) if r_strat.std() > 0 else 0
        ann = r_strat.mean() * 252
        
        results[name] = {'Sharpe': sharpe, 'Ann': ann, 'Sig': sig}
        print(f"Sharpe: {sharpe:.2f}")

    # 6. Final Comparison
    # -------------------
    base_sharpe = r_omni.iloc[start_idx:].mean() / r_omni.iloc[start_idx:].std() * np.sqrt(252)
    
    print("\n" + "="*80)
    print("   FINAL SCOREBOARD: MATH vs GOLDEN RULE")
    print("="*80)
    
    print(f"{'Strategy':<25} | {'Sharpe':<8} | {'Win?'}")
    print("-" * 50)
    
    print(f"{'Golden Omni (Base)':<25} | {base_sharpe:<8.2f} | 🛡️ Benchmark")
    
    for name, stats in results.items():
        win = "✅ YES" if stats['Sharpe'] > base_sharpe else "❌ NO"
        print(f"{name:<25} | {stats['Sharpe']:<8.2f} | {win}")
        
    print("-" * 50)
    
    # Check features info
    print("\nFeature Importance (Boosting):")
    # Quick dirty extraction if last model was boosting
    if 'Boost Reg (Tree)' in models:
        # Re-fit on full data for analysis
        m = models['Boost Reg (Tree)']
        m.fit(features, y_reg)
        impt = pd.Series(m.feature_importances_, index=features.columns).sort_values(ascending=False)
        print(impt.head(5))

if __name__ == "__main__":
    run_regression_shootout()
