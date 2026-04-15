"""
Rebalancing Frequency Analysis for Golden Omni Strategy
========================================================
Tests: Daily, Weekly, Monthly, Quarterly rebalancing
Includes transaction cost impact analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def run_rebalancing_analysis():
    print("=" * 80)
    print("   REBALANCING FREQUENCY DEEP DIVE")
    print("   Testing: Daily vs Weekly vs Monthly vs Quarterly")
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
    
    # 2. Build Signal Components
    # --------------------------
    # Crypto
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
    
    # 3. Define Different Rebalancing Frequencies
    # -------------------------------------------
    frequencies = {
        'Daily': 1,
        'Weekly': 5,
        'Bi-Weekly': 10,
        'Monthly': 21,
        'Quarterly': 63
    }
    
    # Transaction cost per rebalance (round trip: 0.05% = 5bps)
    TX_COST = 0.0005
    
    results = {}
    
    for freq_name, freq_days in frequencies.items():
        # Create rebalanced signal (only updates every N days)
        # Sample the signal every freq_days
        
        is_bull = is_bull_raw.copy()
        is_inflation = is_inflation_raw.copy()
        
        # Resample signals - only change positions every N days
        signal_dates = is_bull.index[::freq_days]
        
        # Forward fill the signal between rebalance dates
        bull_resampled = is_bull.loc[signal_dates].reindex(is_bull.index, method='ffill').shift(1).fillna(False)
        inflation_resampled = is_inflation.loc[signal_dates].reindex(is_inflation.index, method='ffill').shift(1).fillna(False)
        
        # Calculate weights
        w_spy = pd.Series(0.0, index=rets.index)
        w_tlt = pd.Series(0.0, index=rets.index)
        w_xle = pd.Series(0.0, index=rets.index)
        w_gld = pd.Series(0.0, index=rets.index)
        w_crypto = pd.Series(0.0, index=rets.index)
        
        # Bull
        mask_bull = bull_resampled
        w_spy[mask_bull] = 0.45
        w_tlt[mask_bull] = 0.10
        w_gld[mask_bull] = 0.05
        w_crypto[mask_bull] = 0.40
        
        # Deflationary Bear
        mask_def_bear = (~bull_resampled) & (~inflation_resampled)
        w_spy[mask_def_bear] = 0.15
        w_tlt[mask_def_bear] = 0.35
        w_gld[mask_def_bear] = 0.10
        w_crypto[mask_def_bear] = 0.40
        
        # Inflationary Bear
        mask_inf_bear = (~bull_resampled) & inflation_resampled
        w_spy[mask_inf_bear] = 0.15
        w_xle[mask_inf_bear] = 0.35
        w_gld[mask_inf_bear] = 0.10
        w_crypto[mask_inf_bear] = 0.40
        
        # Calculate returns
        r_strat = (
            w_spy * rets['SPY'] +
            w_tlt * rets.get('TLT', 0) +
            w_xle * rets.get('XLE', 0) +
            w_gld * rets.get('GLD', 0) +
            w_crypto * r_crypto +
            missing_crypto
        )
        
        # Calculate turnover and transaction costs
        weights = pd.DataFrame({
            'SPY': w_spy, 'TLT': w_tlt, 'XLE': w_xle, 'GLD': w_gld, 'Crypto': w_crypto
        })
        turnover = weights.diff().abs().sum(axis=1)
        tx_costs = turnover * TX_COST
        
        # Net returns
        r_net = r_strat - tx_costs
        
        # Metrics
        cum = (1 + r_net).cumprod()
        sharpe = r_net.mean() / r_net.std() * np.sqrt(252) if r_net.std() > 0 else 0
        cagr = cum.iloc[-1]**(252/len(cum)) - 1
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        # Count trades
        regime_changes = (weights.diff().abs().sum(axis=1) > 0).sum()
        avg_trades_year = regime_changes / (len(weights) / 252)
        
        total_tx_cost = tx_costs.sum()
        annual_tx_drag = total_tx_cost / (len(r_net) / 252)
        
        results[freq_name] = {
            'Sharpe': sharpe,
            'CAGR': cagr,
            'MaxDD': max_dd,
            'Trades/Year': avg_trades_year,
            'Annual TX Cost': annual_tx_drag
        }
        
    # 4. Print Results
    # ----------------
    print(f"\n{'Frequency':<12} | {'Sharpe':<8} | {'CAGR':<8} | {'MaxDD':<8} | {'Trades/Yr':<10} | {'TX Cost/Yr'}")
    print("-" * 75)
    
    best_sharpe = max(r['Sharpe'] for r in results.values())
    
    for name, stats in results.items():
        marker = "🏆" if stats['Sharpe'] == best_sharpe else "  "
        print(f"{marker}{name:<10} | {stats['Sharpe']:<8.2f} | {stats['CAGR']:<8.1%} | {stats['MaxDD']:<8.1%} | {stats['Trades/Year']:<10.0f} | {stats['Annual TX Cost']:.3%}")
    
    # 5. Analysis
    # -----------
    print("\n" + "=" * 80)
    print("   ANALYSIS")
    print("=" * 80)
    
    # Find optimal
    optimal = max(results.items(), key=lambda x: x[1]['Sharpe'])
    print(f"\n   🏆 OPTIMAL FREQUENCY: {optimal[0]}")
    print(f"   - Sharpe: {optimal[1]['Sharpe']:.2f}")
    print(f"   - CAGR: {optimal[1]['CAGR']:.1%}")
    print(f"   - Trades/Year: {optimal[1]['Trades/Year']:.0f}")
    
    # Compare daily to optimal
    daily = results['Daily']
    print(f"\n   📊 Daily vs {optimal[0]}:")
    print(f"   - Sharpe Improvement: {((optimal[1]['Sharpe'] / daily['Sharpe']) - 1)*100:.1f}%")
    print(f"   - Trade Reduction: {((1 - optimal[1]['Trades/Year'] / daily['Trades/Year']))*100:.0f}%")
    print(f"   - TX Cost Savings: {(daily['Annual TX Cost'] - optimal[1]['Annual TX Cost'])*10000:.1f} bps/year")

if __name__ == "__main__":
    run_rebalancing_analysis()
