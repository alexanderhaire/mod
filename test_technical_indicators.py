
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def run_technical_shootout():
    print("=" * 80)
    print("   TECHNICAL INDICATOR SHOOTOUT")
    print("   Testing: RSI, ATR, MACD vs Baseline")
    print("=" * 80)

    # 1. Fetch Data
    tickers = ['SPY', 'TLT', 'GLD', 'XLE', 'BTC-USD', '^VIX']
    print(f"Fetching data: {tickers}...")
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
             high = data['High'] if 'High' in data.columns.get_level_values(0) else data['Close']
             low = data['Low'] if 'Low' in data.columns.get_level_values(0) else data['Close']
             close = data['Close']
        except:
             prices = data['Close']
             high = data['High']
             low = data['Low']
             close = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        high = data['High']
        low = data['Low']
        close = data['Close']
        
    prices = prices.ffill().dropna(subset=['SPY'])
    rets = prices.pct_change().fillna(0)
    
    # 2. Reconstruct Baseline Omni
    # --------------------------
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
    
    # 3. Create Variants
    results = {}
    
    # Baseline
    results['Golden Omni (Base)'] = r_omni
    
    # Variant 1: RSI Filter
    # Only enter Bull Mode if RSI < 70 (Not Overbought) effectively buying pullbacks?
    # Or avoid Bull Mode if RSI < 30 (Crash)?
    # Let's try: "Risk Off" if RSI > 80 (Extreme Overbought)
    rsi = calculate_rsi(prices['SPY'], 14).shift(1).fillna(50)
    is_bubble = (rsi > 80)
    
    # If bubble, force Defensive Bear mode even if trend is up
    r_rsi = r_omni.copy()
    mask_bubble = is_bull & is_bubble
    r_rsi[mask_bubble] = r_bear[mask_bubble] # Switch to heavy TLT/Cash
    results['RSI Omni (Fade Extremes)'] = r_rsi
    
    # Variant 2: ATR Volatility Sizing
    # Scale exposure inversely to volatility
    # If Vol High -> Lower leverage/weights
    atr = calculate_atr(high['SPY'], low['SPY'], close['SPY']).shift(1).fillna(0)
    atr_norm = atr / close['SPY'].shift(1) # % ATR
    avg_atr = atr_norm.rolling(252).mean()
    
    # Size scaler: Target Vol / Current Vol
    # Cap at 1.0 (no leverage), floor at 0.5
    scaler = (avg_atr / atr_norm).clip(0.5, 1.0).fillna(1.0)
    
    r_atr = r_omni * scaler
    results['ATR Omni (Vol Sizing)'] = r_atr
    
    # Variant 3: MACD Confirmation
    # Only go Bull if Trend > 200 AND MACD > Signal (Momentum Positive)
    macd, macd_sig = calculate_macd(prices['SPY'])
    macd_bull = (macd > macd_sig).shift(1).fillna(False)
    
    # If Bull Logic Met BUT MACD Bearish -> Stay Neutral/Bear
    # i.e., strict requirement for Bull
    is_strict_bull = is_bull & macd_bull
    
    r_macd = pd.Series(0.0, index=rets.index)
    r_macd[is_strict_bull] = r_bull[is_strict_bull]
    # If failed MACD but above 200MA? Maybe just go defensve.
    r_macd[(~is_strict_bull) & (~is_inflation)] = r_bear[(~is_strict_bull) & (~is_inflation)]
    r_macd[(~is_strict_bull) & (is_inflation)] = r_inf[(~is_strict_bull) & (is_inflation)]
    
    results['MACD Omni (Trend Confirm)'] = r_macd
    
    # 4. Compare
    print(f"\n{'Strategy':<25} | {'Sharpe':<8} | {'CAGR':<8} | {'MaxDD':<8}")
    print("-" * 65)
    
    for name, r in results.items():
        if len(r) > 252:
            r = r.iloc[252:] # Skip warmup
            
        sharpe = r.mean() / r.std() * np.sqrt(252)
        cum = (1 + r).cumprod()
        cagr = cum.iloc[-1]**(252/len(cum)) - 1
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        print(f"{name:<25} | {sharpe:>8.2f} | {cagr:>8.1%} | {max_dd:>8.1%}")

if __name__ == "__main__":
    run_technical_shootout()
