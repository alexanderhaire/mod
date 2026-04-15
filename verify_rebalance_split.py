
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def run_is_oos_rebalancing_test():
    print("=" * 80)
    print("   IN-SAMPLE vs OUT-OF-SAMPLE REBALANCING TEST")
    print("   Strategy: Golden Omni")
    print("   Comparison: Daily vs Weekly Rebalancing")
    print("=" * 80)

    # 1. Fetch Data
    tickers = ['SPY', 'TLT', 'GLD', 'XLE', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    print(f"Fetching data: {tickers}...")
    data = yf.download(tickers, start='2010-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna(subset=['SPY'])
    rets = prices.pct_change().fillna(0)
    
    # 2. Build Signal Components (Golden Omni Logic)
    # --------------------------
    
    # Crypto Logic
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True)
    
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    r_crypto = pd.Series(0.0, index=rets.index)
    if avail_alts:
        avg_alt_ret = rets[avail_alts].mean(axis=1).fillna(0)
        btc_mom = btc_p.pct_change(14)
        alt_idx = prices[avail_alts].mean(axis=1)
        alt_mom = alt_idx.pct_change(14)
        is_altseason = (alt_mom > btc_mom).shift(1).fillna(False)
        
        mask_use_alts = is_crypto_safe & is_altseason
        mask_use_btc = is_crypto_safe & ~is_altseason
        
        r_crypto[mask_use_alts] = avg_alt_ret[mask_use_alts]
        if 'BTC-USD' in rets.columns:
            r_crypto[mask_use_btc] = rets['BTC-USD'][mask_use_btc]
    elif 'BTC-USD' in rets.columns:
        r_crypto = rets['BTC-USD'].fillna(0)
        r_crypto[~is_crypto_safe] = 0.0
    
    missing_crypto = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in prices.columns:
        btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
        missing_crypto[~btc_avail] = 0.40 * rets['SPY'][~btc_avail]

    # Regime Signals
    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    is_bull_raw = (spy_p > ma200)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation_raw = (xle_p > ma200_xle) & (~is_bull_raw)

    # 3. Simulate Daily vs Weekly
    # ---------------------------
    # Transaction cost impact (5bps round trip)
    TX_COST = 0.0005 

    results_data = {}

    for freq_name, freq_days in {'Daily': 1, 'Weekly': 5}.items():
        is_bull = is_bull_raw.copy()
        is_inflation = is_inflation_raw.copy()
        
        # Resample logic (sample signal every N days, hold constant between)
        signal_dates = is_bull.index[::freq_days]
        bull_resampled = is_bull.loc[signal_dates].reindex(is_bull.index, method='ffill').shift(1).fillna(False)
        inflation_resampled = is_inflation.loc[signal_dates].reindex(is_inflation.index, method='ffill').shift(1).fillna(False)
        
        # Weights
        w_spy = pd.Series(0.0, index=rets.index)
        w_tlt = pd.Series(0.0, index=rets.index)
        w_xle = pd.Series(0.0, index=rets.index)
        w_gld = pd.Series(0.0, index=rets.index)
        w_crypto = pd.Series(0.0, index=rets.index)
        
        # Strategies Allocations
        # Bull
        mask_bull = bull_resampled
        w_spy[mask_bull] = 0.45; w_tlt[mask_bull] = 0.10; w_gld[mask_bull] = 0.05; w_crypto[mask_bull] = 0.40
        # Deflation Bear
        mask_def_bear = (~bull_resampled) & (~inflation_resampled)
        w_spy[mask_def_bear] = 0.15; w_tlt[mask_def_bear] = 0.35; w_gld[mask_def_bear] = 0.10; w_crypto[mask_def_bear] = 0.40
        # Inflation Bear
        mask_inf_bear = (~bull_resampled) & inflation_resampled
        w_spy[mask_inf_bear] = 0.15; w_xle[mask_inf_bear] = 0.35; w_gld[mask_inf_bear] = 0.10; w_crypto[mask_inf_bear] = 0.40
        
        # Gross Returns
        r_strat = (
            w_spy * rets['SPY'] +
            w_tlt * rets.get('TLT', 0) +
            w_xle * rets.get('XLE', 0) +
            w_gld * rets.get('GLD', 0) +
            w_crypto * r_crypto +
            missing_crypto
        )
        
        # Transaction Costs
        weights = pd.DataFrame({'SPY': w_spy, 'TLT': w_tlt, 'XLE': w_xle, 'GLD': w_gld, 'Cryp': w_crypto})
        turnover = weights.diff().abs().sum(axis=1)
        tx_drag = turnover * TX_COST
        
        r_net = r_strat - tx_drag
        results_data[freq_name] = r_net

    # 4. Split and Analyze
    # --------------------
    SPLIT_DATE = '2020-01-01'
    
    print(f"\nTime Split: {SPLIT_DATE}")
    
    periods = {
        'In-Sample (2010-2019)': (None, SPLIT_DATE),
        'Out-of-Sample (2020-Pres)': (SPLIT_DATE, None)
    }
    
    for pname, (start, end) in periods.items():
        print(f"\n>> {pname}")
        print(f"{'Freq':<10} | {'Sharpe':<8} | {'CAGR':<8} | {'MaxDD':<8} | {'Vol':<8}")
        print("-" * 65)
        
        for freq in ['Daily', 'Weekly']:
            r = results_data[freq]
            
            # Slice
            if start: r = r[r.index >= start]
            if end: r = r[r.index < end]
            
            if len(r) == 0: continue
            
            # Stats
            ann_ret = r.mean() * 252
            vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / vol if vol > 0 else 0
            
            cum = (1+r).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            max_dd = dd.min()
            cagr = cum.iloc[-1]**(252/len(cum)) - 1
            
            marker = ""
            if freq == 'Daily': daily_sharpe = sharpe
            if freq == 'Weekly': 
                marker = "✅" if sharpe > daily_sharpe else "❌"
            
            print(f"{freq:<10} | {sharpe:<8.2f} | {cagr:<8.1%} | {max_dd:<8.1%} | {vol:<8.1%} {marker}")

if __name__ == "__main__":
    run_is_oos_rebalancing_test()
