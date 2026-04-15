
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

warnings.filterwarnings('ignore')

def get_golden_omni_stats():
    """
    Calculates historical stats for Golden Omni using ~20 years of data.
    Returns dict with CAGR, Sharpe, MaxDD.
    """
    print("⏳ Calculating Golden Omni Historical Performance (2005-Present)...")
    
    tickers = ['SPY', 'TLT', 'GLD', 'XLE', 'BTC-USD']
    try:
        data = yf.download(tickers, start='2005-01-01', progress=False)
        try:
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
            prices = data['Close']
            
        prices = prices.ffill()
        rets = prices.pct_change().fillna(0)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    # --- Reconstruct Golden Omni Logic ---
    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    
    is_bull = (spy_p > ma200).shift(1).fillna(False)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation = ((xle_p > ma200_xle) & (~is_bull)).shift(1).fillna(False)
    
    # Crypto Component (Simplified for robust stats)
    btc_ret = rets.get('BTC-USD', pd.Series(0, index=rets.index))
    
    # Allocations
    # Bull: 45% SPY, 10% TLT, 5% GLD, 40% BTC
    r_bull = 0.45*rets['SPY'] + 0.10*rets.get('TLT', 0) + 0.05*rets.get('GLD', 0) + 0.40*btc_ret
    
    # Bear: 15% SPY, 35% TLT, 10% GLD, 40% BTC
    r_bear = 0.15*rets['SPY'] + 0.35*rets.get('TLT', 0) + 0.10*rets.get('GLD', 0) + 0.40*btc_ret
    
    # Inflation: 15% SPY, 35% XLE, 10% GLD, 40% BTC
    r_inf = 0.15*rets['SPY'] + 0.35*rets.get('XLE', 0) + 0.10*rets.get('GLD', 0) + 0.40*btc_ret
    
    r_strat = pd.Series(0.0, index=rets.index)
    r_strat[is_bull] = r_bull[is_bull]
    r_strat[(~is_bull) & (~is_inflation)] = r_bear[(~is_bull) & (~is_inflation)]
    r_strat[(~is_bull) & (is_inflation)] = r_inf[(~is_bull) & (is_inflation)]
    
    # Stats (Annualized)
    # Exclude first 252 days
    if len(r_strat) > 252:
        r_strat = r_strat.iloc[252:]
    
    sharpe = r_strat.mean() / r_strat.std() * np.sqrt(252)
    
    cum = (1 + r_strat).cumprod()
    total_ret = cum.iloc[-1]
    years = len(cum) / 252
    cagr = total_ret**(1/years) - 1
    
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    
    return {
        'Sharpe': sharpe,
        'CAGR': cagr,
        'MaxDD': max_dd
    }

def get_smart_bond_yields(csv_path):
    """
    Analyzes current available "Smart Bonds" from Polymarket CSV.
    """
    print(f"💰 Scanning Smart Bonds from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
        
    # Valid columns?
    if 'Date' not in df.columns or 'Price' not in df.columns:
        print("Missing required columns in CSV")
        return None
        
    # Filter for High Probability "Safe" Bonds
    # Price > 0.94 implies >94% probability roughly
    safe_floor = 0.94
    df_safe = df[df['Price'] >= safe_floor].copy()
    
    if df_safe.empty:
        return {'Count': 0, 'Avg_APY': 0, 'Volume': 0}
        
    # Calculate Yields
    # We assume we are 'today' (2026-01-26 based on user context)
    # If file dates are in future, calculating diff.
    
    # User context says 2026-01-26. 
    current_date = datetime(2026, 1, 26)
    
    def calc_apy(row):
        try:
            # Parse YYYY-MM-DD
            expiry = pd.to_datetime(row['Date'])
            days = (expiry - current_date).days
            
            if days <= 0: days = 0.5 # Expiring today/tomorrow
            
            price = float(row['Price'])
            if price >= 1.0: return 0.0
            
            raw_yield = (1.0 - price) / price
            apy = raw_yield * (365.0 / days)
            return apy
        except:
            return 0.0
            
    df_safe['APY'] = df_safe.apply(calc_apy, axis=1)
    
    # Filter crazy outliers (e.g. typos or 10000% APY due to 0 days)
    df_safe = df_safe[df_safe['APY'] < 50.0] # Cap at 5000% to remove noise
    df_safe = df_safe[df_safe['APY'] > 0.0]  # Only positive yields
    
    avg_apy = df_safe['APY'].mean()
    count = len(df_safe)
    
    # Get top 5 examples
    top_opps = df_safe.sort_values('APY', ascending=False).head(5)
    
    return {
        'Count': count,
        'Avg_APY': avg_apy,
        'Top_Opps': top_opps[['Question', 'Price', 'Date', 'APY']]
    }

def main():
    print("="*60)
    print("⚔️  STRATEGY SHOWDOWN: GOLDEN OMNI vs SMART BOND ⚔️")
    print("="*60)
    
    # 1. Golden Omni
    omni_stats = get_golden_omni_stats()
    
    # 2. Smart Bond
    # Try to find the csv
    csv_path = os.path.join(os.getcwd(), 'data', 'polymarket_real.csv')
    bond_stats = get_smart_bond_yields(csv_path)
    
    print("\n" + "-"*60)
    print("📊 COMPARISON RESULTS")
    print("-"*60)
    
    if omni_stats:
        print(f"🐯 GOLDEN OMNI (Historical 20yr Backtest)")
        print(f"   CAGR (Annual Return): {omni_stats['CAGR']:.1%}")
        print(f"   Sharpe Ratio:         {omni_stats['Sharpe']:.2f}")
        print(f"   Max Drawdown:         {omni_stats['MaxDD']:.1%}")
        print(f"   Risk Profile:         Medium-High (Crypto Volatility)")
    else:
        print("Golden Omni stats failed to calculate.")
        
    print("\n")
    
    if bond_stats:
        print(f"🏛️  SMART BOND (Current Market Snapshot)")
        print(f"   Available 'Safe' Bonds: {bond_stats['Count']}")
        print(f"   Average APY:            {bond_stats['Avg_APY']:.1%}")
        print(f"   Risk Profile:           Low (Assuming Diversified)")
        print("\n   Top Opportunities:")
        for i, row in bond_stats['Top_Opps'].iterrows():
            q = row['Question'][:40] + "..." if len(row['Question']) > 40 else row['Question']
            print(f"   - {q:<45} Price: {row['Price']:.3f}  APY: {row['APY']:.1%}")
    else:
        print("Smart Bond stats failed to calculate.")
        
    print("\n" + "="*60)
    print("🏆 FINAL VERDICT")
    print("="*60)
    
    if omni_stats and bond_stats:
        omni_yield = omni_stats['CAGR']
        bond_yield = bond_stats['Avg_APY']
        
        if bond_yield > omni_yield * 1.2:
            print(f"THE WINNER IS: SMART BOND 🏛️")
            print(f"Reasoning: The risk-free* yield of Smart Bond ({bond_yield:.1%}) significantly outperforms")
            print(f"Golden Omni's historical average ({omni_yield:.1%}). In the current high-yield environment,")
            print("arbitraging prediction markets offers better risk-adjusted returns.")
        elif omni_yield > bond_yield:
            print(f"THE WINNER IS: GOLDEN OMNI 🐯")
            print(f"Reasoning: Smart Bond yields ({bond_yield:.1%}) are insufficient to beat the")
            print(f"growth engine of Golden Omni ({omni_yield:.1%}). Stick to the macro trend.")
        else:
            print(f"THE WINNER IS: TIE / HYBRID ⚖️")
            print(f"Reasoning: Yields are comparable. Use Smart Bond for cash management and Omni for growth.")
    
if __name__ == "__main__":
    main()
