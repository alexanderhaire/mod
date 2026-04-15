"""
Structural & Statistical Alpha Hunt
====================================

Testing 4 market structure hypotheses:
1. Pairs Trading (Stat Arb Mean Reversion)
2. Overnight vs Intraday (The "Night Shift")
3. Turn of Month (Flows)
4. Volatility Carry (Short VIX Risk Premium)

Rigorous testing with Active IR.

RUN: python structural_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA
# =============================================================================

def fetch_structural_data():
    print("📊 Fetching universe (Pairs, Volatility, SPY)...")
    
    # Pairs (Competitors)
    pairs = ['PEP', 'KO', 'CVX', 'XOM', 'MA', 'V', 'HD', 'LOW']
    # Volatility
    vol = ['SVXY']
    # Market
    market = ['SPY']
    
    tickers = pairs + vol + market
    
    # Intraday data for Overnight strategy?
    # yfinance daily data has Open/Close which is sufficient.
    # Start 2018 to capture various regimes.
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    # Need Open and Close for Overnight strategy
    # Access Open/Close directly
    # Note: yfinance multi-ticket download returns MultiIndex columns
    
    return data

# =============================================================================
# STRATEGIES
# =============================================================================

def pairs_trading_strategy(data, lookback=60, entry_z=2.0, exit_z=0.0):
    """
    Hypothesis: Spreads between cointegrated pairs mean revert.
    """
    # Handle DataFrame input (if passed cleaned prices)
    if isinstance(data, pd.DataFrame) and 'Adj Close' not in data.columns and not isinstance(data.columns, pd.MultiIndex):
         prices = data
    else:
         prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
         
    pair_list = [('PEP', 'KO'), ('CVX', 'XOM'), ('MA', 'V'), ('HD', 'LOW')]
    
    returns_df = pd.DataFrame(index=prices.index)
    
    for s1, s2 in pair_list:
        if s1 not in prices.columns or s2 not in prices.columns:
            continue
            
        # Hedge Ratio (Rolling OLS or simple 1:1 for robust test? Let's use Prices Ratio)
        ratio = prices[s1] / prices[s2]
        
        # Z-Score of Ratio
        mean = ratio.rolling(lookback).mean()
        std = ratio.rolling(lookback).std()
        z_score = (ratio - mean) / std
        
        # Signal
        position = pd.Series(0.0, index=prices.index)
        
        # Vectorized signal (Approximation of state loop)
        long_entry = z_score < -entry_z
        short_entry = z_score > entry_z
        exit_cond = abs(z_score) < 0.5 # Close near mean
        
        # Simple Logic: 
        # If Z < -2 -> Long Ratio (Long S1, Short S2)
        # If Z > 2 -> Short Ratio (Short S1, Long S2)
        
        # Calculate Returns of spread
        r1 = prices[s1].pct_change()
        r2 = prices[s2].pct_change()
        
        # Position Logic (Vectorized)
        pos = 0
        pos_list = []
        for z in z_score:
            if z < -entry_z: pos = 1
            elif z > entry_z: pos = -1
            elif abs(z) < 0.5: pos = 0
            pos_list.append(pos)
        
        position = pd.Series(pos_list, index=prices.index).shift(1) # Lag 1 day
        
        # PnL = Position * (Ret S1 - Ret S2)
        pair_ret = position * (r1 - r2) / 2 # Divide by 2 for capital split
        returns_df[f"{s1}_{s2}"] = pair_ret
        
    # Portfolio Return (Equal weight pairs)
    port_ret = returns_df.mean(axis=1).fillna(0)
    return port_ret

def overnight_strategy(data):
    """
    Hypothesis: Buy Close, Sell Open captures the equity premium.
    """
    # Try to extract Open/Close robustly
    try:
        if isinstance(data.columns, pd.MultiIndex):
            opens = data['Open']
            closes = data['Close']
        else:
            opens = data[['Open']] if 'Open' in data.columns else None
            closes = data[['Close']] if 'Close' in data.columns else None
            
        if opens is None or closes is None:
            return pd.Series(0.0, index=data.index)
            
        spy_open = opens['SPY'] if 'SPY' in opens.columns else opens.iloc[:, 0]
        spy_close = closes['SPY'] if 'SPY' in closes.columns else closes.iloc[:, 0]
        
    except Exception as e:
        print(f"   Overnight data error: {e}")
        return pd.Series(0.0, index=data.index)
        
    # Overnight Return: (Open_t / Close_t-1) - 1
    overnight_ret = (spy_open / spy_close.shift(1)) - 1
    return overnight_ret.fillna(0)

def turn_of_month_strategy(data):
    """
    Hypothesis: Long SPY T-1 to T+3 of month.
    """
    # Robust price extraction
    if isinstance(data, pd.DataFrame) and 'SPY' in data.columns and not isinstance(data.columns, pd.MultiIndex):
        prices = data['SPY']
    else:
        try:
             prices = data['Adj Close']['SPY']
        except:
             prices = data['Close']['SPY']
             
    returns = prices.pct_change()
    
    signal = pd.Series(0.0, index=prices.index)
    
    # Identify Month Ends
    # Create DF with Day, Month
    df = pd.DataFrame({'Date': prices.index})
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsMonthStart'] = df['Date'].dt.is_month_start
    df['IsMonthEnd'] = df['Date'].dt.is_month_end
    
    # We need trading days, not calendar days.
    # Logic: Last Trading Day (-1) to 3rd Trading Day (+3)
    
    # Group by Year-Month to find ranks
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # Create trading day index within month
    df['TradeDay'] = df.groupby('YearMonth').cumcount() + 1 # 1st, 2nd...
    
    # Also need reverse count for last day
    df['TradeDayReverse'] = df.groupby('YearMonth')['Date'].transform('count') - df['TradeDay'] 
    
    # Signal: Last day (Reverse=0) or First 3 days (Day <= 3)
    valid_days = (df['TradeDay'] <= 3) | (df['TradeDayReverse'] == 0)
    
    # Map back to index
    # We buy CLOSE of previous day to capture the day's move? 
    # Usually TOM is hold *during* these days.
    # So Signal=1 on these rows.
    
    sig_dates = df[valid_days]['Date']
    signal[signal.index.isin(sig_dates)] = 1
    
    return (signal.shift(1) * returns).fillna(0)

def vol_carry_strategy(prices):
    """
    Hypothesis: Long SVXY (Short VIX) captures variance risk premium.
    Filter: Only when SPY > SMA(200) to avoid crashes.
    """
    if 'SVXY' not in prices.columns or 'SPY' not in prices.columns:
        return None
        
    svxy_ret = prices['SVXY'].pct_change()
    spy = prices['SPY']
    spy_sma = spy.rolling(200).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(200, len(prices)):
        if spy.iloc[i] > spy_sma.iloc[i]:
            signal.iloc[i] = 1 # Regime is safe -> Short Vol
        else:
            signal.iloc[i] = 0 # Regime unsafe -> Cash
            
    return (signal.shift(1) * svxy_ret).fillna(0)

# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def compute_metrics(returns, benchmark_ret=None):
    if len(returns) < 30: return None
    
    # Annualized Metrics
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    ann_ret = returns.mean() * 252
    
    metrics = {
        'Sharpe': sharpe,
        'Ann Ret': ann_ret
    }
    
    if benchmark_ret is not None:
        # Align
        common = returns.index.intersection(benchmark_ret.index)
        act = returns.loc[common] - benchmark_ret.loc[common]
        
        if act.std() > 0:
            ir = act.mean() / act.std() * np.sqrt(252)
            t_stat = act.mean() / (act.std() / np.sqrt(len(act)))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(act)-1))
        else:
            ir = 0; p_val=1
            
        metrics['Active IR'] = ir
        metrics['p-value'] = p_val
    else:
        # For absolute return strategies (Pairs), bench is Cash (0)
        # So Active IR = Sharpe
        metrics['Active IR'] = sharpe
        t_stat = returns.mean() / (returns.std() / np.sqrt(len(returns)))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(returns)-1))
        metrics['p-value'] = p_val
            
    return metrics

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🏛️ STRUCTURAL & STATISTICAL ALPHA HUNT")
    print("="*60)
    
    data = fetch_structural_data()
    
    # Handle YFinance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try to get Adj Close, fallback to Close
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
                
            # For SPY specifically
            if 'SPY' in prices.columns:
                spy_ret = prices['SPY'].pct_change().fillna(0)
            else:
                spy_ret = pd.Series(0.0, index=prices.index)
        except Exception as e:
            print(f"Index error: {e}")
            prices = data
            spy_ret = pd.Series(0.0, index=prices.index)
    else:
        # Flat columns
        prices = data
        if 'SPY' in prices.columns:
            spy_ret = prices['SPY'].pct_change().fillna(0)
            
    print("\n TESTING HYPOTHESES...")
    results = []
    
    # 1. Pairs Trading (Absolute Return)
    # Bench: Cash (0)
    pairs_ret = pairs_trading_strategy(prices)
    m_pairs = compute_metrics(pairs_ret)
    if m_pairs: results.append(({'Name': 'Pairs Trading (Stat Arb)', **m_pairs}))
    
    # 2. Overnight (Bench: SPY Buy Hold)
    # Overnight needs raw data (Open/Close), not just Adj Close
    overnight_ret = overnight_strategy(data)
    m_over = compute_metrics(overnight_ret, spy_ret)
    if m_over: results.append(({'Name': 'Overnight Effect', **m_over}))
    
    # 3. Turn of Month (Bench: SPY Buy Hold)
    tom_ret = turn_of_month_strategy(prices)
    m_tom = compute_metrics(tom_ret, spy_ret)
    if m_tom: results.append(({'Name': 'Turn of Month Flows', **m_tom}))

    # 4. Vol Carry (Bench: SPY - equity risk proxy)
    vol_ret = vol_carry_strategy(prices)
    m_vol = compute_metrics(vol_ret, spy_ret)
    if m_vol: results.append(({'Name': 'Vol Carry (Short VIX)', **m_vol}))
    
    # PRINT RESULTS
    print(f"\n{'Name':<30} {'Sharpe':<8} {'Active IR':<10} {'p-value':<10} {'Sig'}")
    print("-" * 70)
    
    for r in results:
        sig = "✅" if r['p-value'] < 0.05 and r['Active IR'] > 0 else "❌" 
        print(f"{r['Name']:<30} {r['Sharpe']:<8.2f} {r['Active IR']:<10.2f} {r['p-value']:<10.3f} {sig}")


    print("\n" + "=" * 60)
