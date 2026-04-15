
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def run_metals_comparison():
    print("=" * 80)
    print("   GOLDEN OMNI: METALS SHOWDOWN (Gold vs Silver vs Base Metals)")
    print("=" * 80)

    # 1. Fetch Data
    tickers = [
        'SPY', 'TLT', 'GLD', 'SLV', 'DBB', # The Metals
        'XLE', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD' # Core Components
    ]
    print(f"Fetching data for: {tickers}...")
    data = yf.download(tickers, start='2006-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna(subset=['SPY'])
    rets = prices.pct_change().fillna(0)
    
    # 2. Shared Omni Components (Regime & Crypto)
    # Crypto Leg
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True)
    
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    # Simple Crypto Logic for consistency
    if avail_alts:
        avg_alt_ret = rets[avail_alts].mean(axis=1).fillna(0)
        btc_mom = btc_p.pct_change(14)
        alt_idx = prices[avail_alts].mean(axis=1)
        alt_mom = alt_idx.pct_change(14)
        is_altseason = (alt_mom > btc_mom).shift(1).fillna(False)
    else:
        is_altseason = pd.Series(False, index=prices.index)
        avg_alt_ret = pd.Series(0.0, index=prices.index)
        
    mask_use_alts = is_crypto_safe & is_altseason
    mask_use_btc = is_crypto_safe & ~is_altseason
    
    r_crypto = pd.Series(0.0, index=rets.index)
    r_crypto[mask_use_alts] = avg_alt_ret[mask_use_alts]
    
    if 'BTC-USD' in rets.columns:
        r_crypto[mask_use_btc] = rets['BTC-USD'][mask_use_btc]
        
    # Correction for missing BTC history (allocating to SPY)
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

    # 3. Define Variants
    # Function to calculate Omni returns with a specific Metal ticker
    def calc_omni(metal_ticker):
        if metal_ticker not in rets.columns:
            return pd.Series(0.0, index=rets.index)
            
        r_metal = rets[metal_ticker]
        
        # Bull: 45 SPY, 10 TLT, 5 Metal, 40 Crypto
        r_bull = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.05 * r_metal + 0.40 * r_crypto + missing_crypto)
        
        # Bear (Deflation): 15 SPY, 35 TLT, 10 Metal, 40 Crypto
        r_bear = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.10 * r_metal + 0.40 * r_crypto + missing_crypto)
        
        # Bear (Inflation): 15 SPY, 35 XLE, 10 Metal, 40 Crypto
        r_inf = (0.15 * rets['SPY'] + 0.35 * rets.get('XLE', 0) + 0.10 * r_metal + 0.40 * r_crypto + missing_crypto)
        
        r_strat = pd.Series(0.0, index=rets.index)
        r_strat[is_bull] = r_bull[is_bull]
        r_strat[(~is_bull) & (~is_inflation)] = r_bear[(~is_bull) & (~is_inflation)]
        r_strat[(~is_bull) & (is_inflation)] = r_inf[(~is_bull) & (is_inflation)]
        
        return r_strat

    # Run Variants
    results = {}
    
    # 1. Baseline (Gold)
    results['Golden Omni (Gold)'] = calc_omni('GLD')
    
    # 2. Silver Omni
    results['Silver Omni (SLV)'] = calc_omni('SLV')
    
    # 3. Base Metals Omni (Industrial: Copper, Zinc, Aluminum)
    results['Industrial Omni (DBB)'] = calc_omni('DBB')
    
    # 4. Mixed Omni (50/50 Gold/Silver)
    if 'GLD' in rets.columns and 'SLV' in rets.columns:
        r_mixed_metal = 0.5 * rets['GLD'] + 0.5 * rets['SLV']
        rets['MIXED_METAL'] = r_mixed_metal
        results['Mixed Omni (Gold+Silver)'] = calc_omni('MIXED_METAL')

    # 4. Compare Results
    print(f"\n{'Strategy':<25} | {'Sharpe':<8} | {'CAGR':<8} | {'MaxDD':<8} | {'Vol':<8}")
    print("-" * 75)
    
    for name, r in results.items():
        # Trim to start only when we have data for that variant
        # (SLV started later than GLD in some datasets, check first valid)
        r = r.replace(0, np.nan).dropna() # Simple trim to valid period
        
        if r.empty:
            continue
            
        sharpe = r.mean() / r.std() * np.sqrt(252)
        cum = (1 + r).cumprod()
        cagr = cum.iloc[-1]**(252/len(cum)) - 1
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        vol = r.std() * np.sqrt(252)
        
        print(f"{name:<25} | {sharpe:>8.2f} | {cagr:>8.1%} | {max_dd:>8.1%} | {vol:>8.1%}")

if __name__ == "__main__":
    run_metals_comparison()
