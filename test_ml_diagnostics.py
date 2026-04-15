"""
ML Ensemble Post-Mortem: Why did it fail?
=========================================

Diagnostic script to analyze the 'Brain' of the strategy.
1. Feature Importance: What is the model looking at?
2. Predictive Power (IC): Do the predictions actually predict returns?
3. Directional Accuracy: Is it better than a coin flip?

This gives the user closure on whether the *idea* was bad or just the *calibration*.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import spearmanr
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig

# =============================================================================
# DATA
# =============================================================================

# Test on the "Champion" universe where it supposedly "worked" (but failed selection bias)
TICKERS = ['EWJ', 'EWY', 'AAXJ', 'GLD', 'TLT']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(TICKERS, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

# =============================================================================
# FEATURE ENGINEERING REPLICATION
# =============================================================================

def create_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate the features used by the strategy."""
    returns = prices.pct_change()
    
    features = pd.DataFrame(index=prices.index)
    targets = pd.DataFrame(index=prices.index)
    
    # Create features for each asset
    for ticker in prices.columns:
        # Momentum
        features[f'{ticker}_mom_1m'] = returns[ticker].rolling(20).mean()
        features[f'{ticker}_mom_3m'] = returns[ticker].rolling(60).mean()
        features[f'{ticker}_mom_6m'] = returns[ticker].rolling(120).mean()
        
        # Volatility
        features[f'{ticker}_vol_1m'] = returns[ticker].rolling(20).std()
        
        # RSI-like (simplified)
        delta = returns[ticker]
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        # Target: Forward 5-day return
        targets[f'{ticker}_target'] = returns[ticker].shift(-5).rolling(5).sum()
        
    return features.dropna(), targets.dropna()

# =============================================================================
# DIAGNOSTICS
# =============================================================================

def run_diagnostics():
    print("=" * 80)
    print("   ML ENSEMBLE POST-MORTEM DIAGNOSTICS")
    print("   Investigating the 'Brain' of the strategy")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    features, targets = create_features(prices)
    
    # Align
    common_index = features.index.intersection(targets.index)
    features = features.loc[common_index]
    targets = targets.loc[common_index]
    
    print(f"\n   Data Points: {len(features)}")
    print(f"   Features: {features.shape[1]}")
    
    # Train/Test Split
    split = int(len(features) * 0.7)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = targets.iloc[:split], targets.iloc[split:]
    
    results = []
    
    print("\n   Analyzing per-asset predictive power...")
    print(f"   {'Asset':<10} {'IC (Corr)':>10} {'Accuracy':>10} {'Feature Importances (Top 2)'}")
    print("   " + "-" * 70)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    
    avg_ic = 0
    avg_acc = 0
    
    for ticker in prices.columns:
        # Get asset specific features
        asset_feats = [c for c in features.columns if c.startswith(ticker)]
        target_col = f'{ticker}_target'
        
        if not asset_feats: continue
        
        X_tr = X_train[asset_feats]
        y_tr = y_train[target_col]
        X_te = X_test[asset_feats]
        y_te = y_test[target_col]
        
        # Train
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        # 1. Information Coefficient (IC) - Correlation between prediction and actual
        ic, _ = spearmanr(preds, y_te)
        
        # 2. Directional Accuracy
        direction_pred = np.sign(preds)
        direction_actual = np.sign(y_te)
        accuracy = accuracy_score(direction_actual, direction_pred)
        
        # 3. Feature Importance
        importances = pd.Series(model.feature_importances_, index=asset_feats).sort_values(ascending=False)
        top_feats = ", ".join(importances.head(2).index.tolist())
        top_feats = top_feats.replace(f'{ticker}_', '') # Clean name
        
        print(f"   {ticker:<10} {ic:>10.3f} {accuracy:>10.1%}   {top_feats}")
        
        results.append({'ticker': ticker, 'ic': ic, 'accuracy': accuracy})
        avg_ic += ic
        avg_acc += accuracy
    
    avg_ic /= len(prices.columns)
    avg_acc /= len(prices.columns)
    
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC CONCLUSION")
    print("=" * 80)
    
    print(f"\n   Average Predictive Correlation (IC): {avg_ic:.3f}")
    if avg_ic < 0.02:
        print("   ❌ PREDICTIVE POWER IS NEAR ZERO (Noise). Good algos usually have IC > 0.05.")
    else:
        print("   ✅ Some predictive power exists.")
        
    print(f"   Average Directional Accuracy: {avg_acc:.1%}")
    if avg_acc < 0.52:
        print("   ❌ ACCURACY IS COIN FLIP (50%). No edge in predicting direction.")
    else:
        print("   ✅ Slight edge in direction.")
        
    print("\n   WHY IT FAILED:")
    if avg_ic < 0.02 and avg_acc < 0.52:
        print("   The input features (Momentum, RSI, Volatility) simply DOES NOT PREDICT")
        print("   future returns for these assets in this regime. Garbage In, Garbage Out.")
        print("   The ML assumed these standard indicators work, but the market has evolved.")
    elif avg_ic > 0.05:
        print("   The model actually has some insights! The failure might be in sizing/risk management.")
    else:
        print("   The edge is too thin to overcome transaction costs and noise.")

    return results

if __name__ == "__main__":
    run_diagnostics()
