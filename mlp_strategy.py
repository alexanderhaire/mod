"""
Advanced Multi-Layer Perceptron Strategy
==========================================

A deep neural network approach to predict optimal allocations.

Architecture:
- Input: 50+ engineered features (rolling windows, momentum, volatility regimes, etc.)
- Hidden Layers: 256 -> 128 -> 64 -> 32 (with BatchNorm, Dropout)
- Output: Allocation weights across assets

Training:
- Walk-forward validation (no look-ahead bias)
- Retrain every 63 days (quarterly)
- Target: Next 5-day returns

This is the "throw everything at the wall" ML approach to compare against
hand-crafted strategies like Golden Omni.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create 50+ features for the MLP to learn from.
    These are the same signals our hand-crafted strategies use,
    but we let the MLP learn how to combine them.
    """
    features = pd.DataFrame(index=prices.index)
    
    # Core assets
    spy = prices.get('SPY', pd.Series(np.nan, index=prices.index))
    tlt = prices.get('TLT', pd.Series(np.nan, index=prices.index))
    gld = prices.get('GLD', pd.Series(np.nan, index=prices.index))
    btc = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    eth = prices.get('ETH-USD', pd.Series(np.nan, index=prices.index))
    vix = prices.get('^VIX', pd.Series(20, index=prices.index))
    
    # =========================================================================
    # TREND FEATURES (What Golden Omni uses)
    # =========================================================================
    for lookback in [20, 50, 100, 200]:
        # Price vs MA
        features[f'spy_vs_ma{lookback}'] = (spy / spy.rolling(lookback).mean() - 1)
        features[f'tlt_vs_ma{lookback}'] = (tlt / tlt.rolling(lookback).mean() - 1)
        features[f'gld_vs_ma{lookback}'] = (gld / gld.rolling(lookback).mean() - 1)
        
        # BTC/ETH (handle NaNs for early dates)
        btc_ma = btc.rolling(lookback).mean()
        features[f'btc_vs_ma{lookback}'] = (btc / btc_ma - 1).fillna(0)
        
    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================
    for period in [5, 10, 21, 63, 126, 252]:
        features[f'spy_mom_{period}d'] = spy.pct_change(period)
        features[f'tlt_mom_{period}d'] = tlt.pct_change(period)
        features[f'btc_mom_{period}d'] = btc.pct_change(period).fillna(0)
        
    # Cross-asset momentum
    features['spy_tlt_spread_mom'] = features['spy_mom_21d'] - features['tlt_mom_21d']
    features['btc_spy_spread_mom'] = features['btc_mom_21d'] - features['spy_mom_21d']
    
    # =========================================================================
    # VOLATILITY FEATURES (What Ultimate Strategy uses)
    # =========================================================================
    for window in [10, 21, 63]:
        features[f'spy_vol_{window}d'] = spy.pct_change().rolling(window).std() * np.sqrt(252)
        features[f'btc_vol_{window}d'] = btc.pct_change().rolling(window).std().fillna(0) * np.sqrt(365)
        
    # VIX signals
    features['vix_level'] = vix
    features['vix_vs_ma20'] = (vix / vix.rolling(20).mean() - 1)
    features['vix_vs_ma50'] = (vix / vix.rolling(50).mean() - 1)
    features['vix_change_5d'] = vix.pct_change(5)
    
    # Realized vs Implied Vol spread (VRP)
    features['vrp'] = (spy.pct_change().rolling(21).std() * np.sqrt(252) * 100) - vix
    
    # =========================================================================
    # CRYPTO-SPECIFIC FEATURES (Altseason, Flipper logic)
    # =========================================================================
    if 'ETH-USD' in prices.columns and 'BTC-USD' in prices.columns:
        eth_btc_ratio = eth / btc
        features['eth_btc_ratio'] = eth_btc_ratio.fillna(0)
        features['eth_btc_zscore'] = (
            (eth_btc_ratio - eth_btc_ratio.rolling(90).mean()) / 
            eth_btc_ratio.rolling(90).std()
        ).fillna(0)
        
        # Altseason signal
        btc_mom_14 = btc.pct_change(14)
        eth_mom_14 = eth.pct_change(14)
        features['altseason_signal'] = (eth_mom_14 > btc_mom_14).astype(float).fillna(0)
        
    # =========================================================================
    # REGIME FEATURES (Binary signals as continuous)
    # =========================================================================
    features['bull_regime'] = (spy > spy.rolling(200).mean()).astype(float)
    features['fear_regime'] = (vix > vix.rolling(20).mean()).astype(float)
    
    # Inflation proxy (XLE)
    xle = prices.get('XLE', spy)
    features['inflation_regime'] = (xle > xle.rolling(200).mean()).astype(float)
    
    # =========================================================================
    # CORRELATION FEATURES
    # =========================================================================
    spy_ret = spy.pct_change()
    tlt_ret = tlt.pct_change()
    btc_ret = btc.pct_change().fillna(0)
    
    features['spy_tlt_corr_21d'] = spy_ret.rolling(21).corr(tlt_ret)
    features['spy_btc_corr_21d'] = spy_ret.rolling(21).corr(btc_ret).fillna(0)
    
    # =========================================================================
    # DAY-OF-WEEK (What Crypto Comp uses)
    # =========================================================================
    features['dow_monday'] = (features.index.dayofweek == 0).astype(float)
    features['dow_wednesday'] = (features.index.dayofweek == 2).astype(float)
    features['dow_thursday'] = (features.index.dayofweek == 3).astype(float)
    features['dow_friday'] = (features.index.dayofweek == 4).astype(float)
    
    # =========================================================================
    # TIME FEATURES (Seasonality as Cyclical Features)
    # =========================================================================
    
    # Month of Year (Seasonality: Sell in May, Santa Rally, etc.)
    features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)
    
    # Day of Month (Paycheck/Inflow effects, Turn-of-month)
    features['dom_sin'] = np.sin(2 * np.pi * features.index.day / 31)
    features['dom_cos'] = np.cos(2 * np.pi * features.index.day / 31)
    
    # Day of Year (Long annual cycles)
    features['doy_sin'] = np.sin(2 * np.pi * features.index.dayofyear / 365)
    features['doy_cos'] = np.cos(2 * np.pi * features.index.dayofyear / 365)
    
    # =========================================================================
    # INTRADAY / OVERNIGHT / MARKET OPEN EFFECTS 
    # =========================================================================
    # Since we use DAILY data, we approximate intraday effects:
    # 1. Overnight Return (Close-to-Open gap) - requires Open prices
    # 2. Gap from Weekend/Holiday (days since last trading day)
    # 3. Post-holiday effect (first trading day after multi-day break)
    
    # Days since last trading day (detects weekend/holiday gaps)
    trading_dates = pd.Series(features.index)
    days_gap = trading_dates.diff().dt.days.fillna(1).values
    features['days_since_last_trade'] = days_gap
    
    # Post-weekend effect (Monday or first day after holiday)
    features['is_post_weekend'] = (days_gap >= 3).astype(float)
    features['is_post_holiday'] = (days_gap >= 4).astype(float)  # Long weekend/holiday
    
    # Week number within month (Turn-of-month effects)
    features['week_of_month'] = ((features.index.day - 1) // 7 + 1) / 5
    
    # First/Last trading days of month (institutional rebalancing)
    features['is_month_start'] = (features.index.day <= 3).astype(float)
    features['is_month_end'] = (features.index.day >= 28).astype(float)
    
    # Quarter boundaries (big rebalancing days)
    features['is_quarter_end'] = ((features.index.month % 3 == 0) & (features.index.day >= 28)).astype(float)
    
    # =========================================================================
    # US BANK HOLIDAYS (Approximate - fixed date holidays)
    # =========================================================================
    # Major US market holidays that affect trading behavior
    month = features.index.month
    day = features.index.day
    
    # New Year's Day effect (Jan 1-3)
    features['near_new_year'] = ((month == 1) & (day <= 5)).astype(float)
    
    # MLK Day effect (3rd Monday of Jan, approx Jan 15-21)
    features['near_mlk_day'] = ((month == 1) & (day >= 15) & (day <= 21)).astype(float)
    
    # Presidents Day effect (3rd Monday of Feb, approx Feb 15-21)
    features['near_presidents_day'] = ((month == 2) & (day >= 15) & (day <= 21)).astype(float)
    
    # Good Friday / Easter effect (variable, but usually late March/April)
    features['near_easter'] = ((month == 3) & (day >= 20) | (month == 4) & (day <= 20)).astype(float)
    
    # Memorial Day effect (Last Monday of May, approx May 25-31)
    features['near_memorial_day'] = ((month == 5) & (day >= 25)).astype(float)
    
    # Independence Day effect (July 4)
    features['near_july_4th'] = ((month == 7) & (day >= 1) & (day <= 7)).astype(float)
    
    # Labor Day effect (1st Monday of Sept, approx Sept 1-7)
    features['near_labor_day'] = ((month == 9) & (day <= 7)).astype(float)
    
    # Thanksgiving effect (4th Thursday of Nov, approx Nov 22-28)
    features['near_thanksgiving'] = ((month == 11) & (day >= 22) & (day <= 28)).astype(float)
    
    # Christmas/Year-end effect (Dec 20 - Dec 31, "Santa Rally")
    features['near_christmas'] = ((month == 12) & (day >= 20)).astype(float)
    
    # General "Holiday Season" (Nov 15 - Jan 5)
    features['holiday_season'] = (((month == 11) & (day >= 15)) | (month == 12) | ((month == 1) & (day <= 5))).astype(float)
    
    # =========================================================================
    # MARKET SESSION PROXY (Using VIX behavior)
    # =========================================================================
    # VIX tends to be elevated at open, drops through day
    # We use VIX changes as a proxy for "time of day" stress patterns
    vix_intraday_proxy = vix.pct_change()
    features['vix_daily_change'] = vix_intraday_proxy.fillna(0)
    features['vix_opening_stress'] = (vix_intraday_proxy > 0.05).astype(float).fillna(0)  # Big VIX spike = stressed open
    
    # =========================================================================
    # CLEAN UP
    # =========================================================================
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    return features


def calculate_mlp_returns(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """
    Train and run the MLP strategy with walk-forward validation.
    
    This is the key function - it trains the MLP on historical data
    and generates trading signals WITHOUT look-ahead bias.
    """
    if not HAS_SKLEARN:
        print("sklearn not available, returning SPY returns")
        return rets.get('SPY', pd.Series(0, index=rets.index))
    
    # Target: 5-day forward return of a simple SPY/TLT/BTC blend
    # The MLP will learn to predict which blend is optimal
    target_assets = ['SPY', 'TLT', 'BTC-USD']
    available = [a for a in target_assets if a in rets.columns]
    
    if len(available) < 2:
        return rets.get('SPY', pd.Series(0, index=rets.index))
    
    # Simple target: predict optimal weight for SPY vs TLT
    # We'll use sign of next 5-day diff between SPY and TLT
    spy_fwd = rets['SPY'].rolling(5).sum().shift(-5)
    tlt_fwd = rets.get('TLT', rets['SPY']).rolling(5).sum().shift(-5)
    
    # Target: 1 if SPY beats TLT next 5 days, 0 otherwise (regression target)
    target = (spy_fwd - tlt_fwd)
    
    # Features
    features = engineer_features(prices)
    
    # Align
    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    
    # Walk-forward training
    # Train on 2 years, predict next quarter, retrain
    train_window = 504  # ~2 years
    retrain_freq = 63   # quarterly
    
    predictions = pd.Series(index=rets.index, dtype=float)
    
    # MLP architecture - DEEP
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),  # Deep architecture
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2 regularization
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )
    
    scaler = StandardScaler()
    
    # Walk-forward loop
    for i in range(train_window, len(X), retrain_freq):
        # Training data
        train_idx = X.index[:i]
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        
        # Remove any NaN targets
        valid = y_train.notna()
        X_train = X_train[valid]
        y_train = y_train[valid]
        
        if len(X_train) < 100:
            continue
            
        # Scale features
        X_scaled = scaler.fit_transform(X_train)
        
        try:
            # Train
            mlp.fit(X_scaled, y_train)
            
            # Predict next quarter
            end_idx = min(i + retrain_freq, len(X))
            pred_idx = X.index[i:end_idx]
            X_pred = X.loc[pred_idx]
            X_pred_scaled = scaler.transform(X_pred)
            
            preds = mlp.predict(X_pred_scaled)
            predictions.loc[pred_idx] = preds
            
        except Exception as e:
            # If training fails, continue
            continue
    
    # Convert predictions to returns
    # Positive prediction = overweight SPY, negative = overweight TLT
    predictions = predictions.fillna(0)
    
    # Normalize predictions to [-1, 1] for signal strength
    pred_std = predictions.rolling(252).std()
    pred_std = pred_std.replace(0, 1).fillna(1)
    signal = (predictions / pred_std).clip(-2, 2) / 2  # Now in [-1, 1]
    
    # Convert to weights
    # signal = 1 -> 100% SPY, signal = -1 -> 100% TLT, signal = 0 -> 50/50
    w_spy = (signal + 1) / 2  # Maps [-1,1] to [0,1]
    w_tlt = 1 - w_spy
    
    # Add crypto allocation when available
    btc_available = (prices.get('BTC-USD', pd.Series(0, index=prices.index)) > 0)
    
    # Reduce SPY/TLT weights to add BTC when available
    btc_weight = 0.20  # 20% to crypto when available
    w_spy_adj = w_spy * (1 - btc_weight * btc_available.astype(float))
    w_tlt_adj = w_tlt * (1 - btc_weight * btc_available.astype(float))
    w_btc = btc_weight * btc_available.astype(float)
    
    # Shift weights for T+1 execution
    w_spy_adj = w_spy_adj.shift(1).fillna(0.5)
    w_tlt_adj = w_tlt_adj.shift(1).fillna(0.5)
    w_btc = w_btc.shift(1).fillna(0)
    
    # Calculate returns
    r_mlp = (
        w_spy_adj * rets['SPY'] + 
        w_tlt_adj * rets.get('TLT', 0) + 
        w_btc * rets.get('BTC-USD', pd.Series(0, index=rets.index)).fillna(0)
    )
    
    return r_mlp


def get_mlp_equity_curve(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """
    Main entry point - returns the equity curve for the MLP strategy.
    """
    r_mlp = calculate_mlp_returns(prices, rets)
    equity = (1 + r_mlp).cumprod()
    return equity


if __name__ == "__main__":
    # Test the MLP strategy
    import yfinance as yf
    
    print("Downloading test data...")
    tickers = ['SPY', 'TLT', 'GLD', 'BTC-USD', 'ETH-USD', '^VIX', 'XLE']
    prices = yf.download(tickers, start='2015-01-01', progress=False)['Adj Close']
    prices = prices.ffill().dropna(subset=['SPY'])
    rets = prices.pct_change().fillna(0)
    
    print("Training MLP strategy...")
    equity = get_mlp_equity_curve(prices, rets)
    
    # Calculate stats
    r = equity.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    
    print(f"\n=== MLP Strategy Results ===")
    print(f"Total Return: {total_return:.1%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Final Value: ${equity.iloc[-1]:.2f}")
