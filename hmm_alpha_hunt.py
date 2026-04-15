"""
HMM Alpha Hunt (Hidden Markov / Gaussian Mixture)
=================================================

Unsupervised Learning to detect "Hidden Market Regimes".
Theory: Market returns follow a Mixture Distribution.
Model: GaussianMixture (GMM) with 2 or 3 components.

Strategy:
1. Fit GMM on [Returns, Volatility].
2. Identify "Bull State" (High Mean, Low Vol).
3. Long SPY when Prob(Bull State) > 50%.
4. Cash/Short otherwise.

RUN: python hmm_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA
# =============================================================================

def fetch_data():
    print("🔮 Fetching SPY Data for HMM...")
    data = yf.download("SPY", start='2010-01-01', progress=False)
    
    # Robust extraction
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days")
    return prices

# =============================================================================
# 2. FEATURE ENGINEERING & HMM
# =============================================================================

def train_hmm(prices, n_components=2):
    returns = prices.pct_change().dropna()
    vol = returns.rolling(10).std().dropna() # Fast vol
    
    # Align
    df = pd.concat([returns, vol], axis=1).dropna()
    df.columns = ['Ret', 'Vol']
    
    # X used for training
    X = df.values
    
    print(f"🤖 Training GMM with {n_components} components...")
    model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    model.fit(X)
    
    # Predict States
    states = model.predict(X)
    df['State'] = states
    
    # Analyze States to identify "Bull"
    print("\n🧐 Analyzing Regimes:")
    bull_state = -1
    best_sharpe = -999
    
    for i in range(n_components):
        state_df = df[df['State'] == i]
        mu = state_df['Ret'].mean() * 252
        sigma = state_df['Ret'].std() * np.sqrt(252)
        sharpe = mu / sigma if sigma > 0 else 0
        count = len(state_df)
        print(f"   State {i}: Ann Ret {mu:.1%} | Vol {sigma:.1%} | Sharpe {sharpe:.2f} | Days {count}")
        
        # We want the state with highest Sharpe (or highest positive return)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            bull_state = i
            
    print(f"   👉 Selected Bull State: {bull_state}")
    
    return model, bull_state, df

# =============================================================================
# 3. BACKTEST (Walk Forward Simulation approximation)
# =============================================================================

def backtest_hmm(prices):
    # Train heavily on past, predict future?
    # Simple In-Sample test first to see if Regimes even EXIST.
    # If In-Sample fails, Out-Sample definitely fails.
    
    model, bull_node, df = train_hmm(prices, n_components=2)
    
    # Strategy: Long if State == BullNode
    # Note: 'State' at index t uses Ret(t). This is LOOKAHEAD.
    # We must predict State(t+1) using info at t?
    # No, GMM clusters observations.
    # We need to predict "Current Regime".
    # IF we are in Bull Regime today, we assume we stay in it tomorrow (Persistence).
    
    signal = (df['State'] == bull_node).astype(int)
    
    # Shift signal to trade tomorrow
    strat_ret = signal.shift(1) * df['Ret']
    strat_ret = strat_ret.dropna()
    
    # Buy Hold
    bh_ret = df['Ret']
    
    # Metrics
    def get_stats(r):
        ann_ret = r.mean() * 252
        sharpe = r.mean() / r.std() * np.sqrt(252)
        return ann_ret, sharpe
        
    s_bh, sharpe_bh = get_stats(bh_ret)
    s_strat, sharpe_strat = get_stats(strat_ret)
    
    print("\n📊 HMM STRATEGY RESULTS (In-Sample):")
    print(f"   Buy & Hold: Return {s_bh:.1%} | Sharpe {sharpe_bh:.2f}")
    print(f"   HMM Strategy: Return {s_strat:.1%} | Sharpe {sharpe_strat:.2f}")
    
    if sharpe_strat > sharpe_bh + 0.2:
        print("   ✅ EDGE FOUND: Regimes are persistent.")
    else:
        print("   ❌ NO EDGE: Regimes are random/noisy.")

if __name__ == "__main__":
    print("="*60)
    print("🧠 HMM ALPHA HUNT (REGIME SWITCHING)")
    print("="*60)
    
    prices = fetch_data()
    backtest_hmm(prices)
    
    print("\n" + "=" * 60)
