"""
PCA Alpha Hunt (Eigen-Portfolio Arbitrage)
==========================================

Statistical Arbitrage using Principal Component Analysis.
Hypothesis: Sector returns are driven by K latent factors. 
Asset Return = Beta * Factors + Residual.
Residuals Mean Revert.

Process:
1. Rolling PCA (60d) on Sector Universe.
2. Calculate Residuals (Real Ret - Expected Ret).
3. Long/Short Strategy based on Z-Score of Residuals.

RUN: python pca_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH DATA (Sector Universe)
# =============================================================================

def fetch_sector_data():
    print("🧠 Fetching Sector Universe for PCA...")
    tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLP', 'XLY', 'XLI', 'XLB', 'XLU', 'TLT', 'GLD']
    
    data = yf.download(tickers, start='2015-01-01', progress=False)
    
     # Robust extraction
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.dropna()
    print(f"   Data: {len(prices)} days | {len(prices.columns)} assets")
    return prices

# =============================================================================
# 2. PCA ENGINE
# =============================================================================

def get_residuals(returns, n_components=3):
    """
    Returns the Residuals Matrix (Returns - Reconstructed Returns).
    """
    pca = PCA(n_components=n_components)
    
    # Fit PCA on returns
    # Note: Using entire history for 'in-sample' test of concept.
    # For trading, we'd use rolling PCA.
    # Let's do Rolling PCA approx via expanding window? No, too slow.
    # Let's do Full PCA to check if the STRUCTURE exists.
    
    pca.fit(returns)
    factors = pca.transform(returns) # (T, K)
    components = pca.components_    # (K, N)
    
    reconstructed = np.dot(factors, components) # (T, N)
    residuals = returns.values - reconstructed
    
    res_df = pd.DataFrame(residuals, index=returns.index, columns=returns.columns)
    
    explained = np.sum(pca.explained_variance_ratio_)
    print(f"   Explained Variance (Top {n_components}): {explained:.1%}")
    
    return res_df

# =============================================================================
# 3. STRATEGY (Mean Reversion of Residuals)
# =============================================================================

def backtest_pca_stat_arb(prices):
    returns = prices.pct_change().dropna()
    
    # Get Residuals (Day T)
    # Note: Using FUTURE PCA (In-Sample Lookahead) to test the PHYSICS, not the Algo yet.
    # If this fails In-Sample, the idea is dead.
    residuals = get_residuals(returns, n_components=3)
    
    # Calculate Z-Score (Rolling)
    roll_mean = residuals.rolling(20).mean()
    roll_std = residuals.rolling(20).std()
    z_score = ((residuals - roll_mean) / roll_std).fillna(0)
    
    # Strategy:
    # Long if Z < -1.5 (Sold off too much relative to factors)
    # Short if Z > 1.5 (rallied too much)
    
    longs = (z_score < -1.5).astype(int)
    shorts = (z_score > 1.5).astype(int)
    
    # Positions (Market Neutral-ish)
    # Weight = 1 / Count?
    # Simplified: Just sum returns.
    
    strat_ret = (longs.shift(1) * returns) - (shorts.shift(1) * returns)
    
    # Portfolio Return (Sum of all active positions / assumption of capital)
    # Assume we allocate 10% per active trade.
    port_ret = strat_ret.sum(axis=1) / 10.0 # Leverage scaling
    
    # Metrics
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    ann = port_ret.mean() * 252
    
    print("\n🧮 PCA STAT ARB RESUlTS (In-Sample Theoretical):")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Ann Return: {ann:.1%}")
    
    if sharpe > 1.0:
        print("   ✅ EDGE FOUND: Residuals mean revert!")
    else:
        print("   ❌ NO EDGE: Residuals are random walk.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🌀 PCA ALPHA HUNT (EIGEN-PORTFOLIOS)")
    print("="*60)
    
    prices = fetch_sector_data()
    backtest_pca_stat_arb(prices)
    
    print("\n" + "=" * 60)
