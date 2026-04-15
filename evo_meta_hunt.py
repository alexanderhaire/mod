"""
Evolutionary Meta-Param Optimization Hunt
=========================================

Using Genetic Algorithms (GA) to find the "Golden Parameter Set".
Optimizing the entire System (Ultimate + HRP + Dollar) simultaneously.

Genes:
1. Ult_VIX_SMA (Trend trigger for Ulltimate)
2. HRP_Lookback (Volatility calc window)
3. UUP_Fast (Dollar Trend Entry)
4. UUP_Slow (Dollar Trend Baseline)
5. Alloc_Split (Balance between Ult and HRP)
6. UUP_Size (How much allocation to UUP when active)

Method:
- In-Sample (Train): 2010-2019
- Out-of-Sample (Test): 2020-2024
- Objective: Maximize Sharpe Ratio.

RUN: python evo_meta_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA
# =============================================================================

def fetch_data():
    print("🧬 Fetching Evolutionary Data...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', # Trad
        'UUP', # Dollar
        'BTC-USD', # Crypto
        '^VIX' # Macro
    ]
    data = yf.download(tickers, start='2010-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    
    cols_to_drop = [c for c in ['^VIX'] if c in prices.columns]
    prices = prices.drop(columns=cols_to_drop).ffill().dropna()
    
    if vix is not None: vix = vix.reindex(prices.index).ffill()
    print(f"   Data: {len(prices)} days")
    return prices, vix

# =============================================================================
# 2. SYSTEM ENGINE (The Phenotype)
# =============================================================================

def run_system_logic(prices, vix, params):
    """
    params: [vix_ma, hrp_look, uup_fast, uup_slow, alloc_split, uup_size]
    """
    vix_ma_win = int(params[0])
    hrp_look = int(params[1])
    uup_fast = int(params[2])
    uup_slow = int(params[3])
    alloc_split = params[4] # 0 = 100% Ult, 1 = 100% HRP
    uup_size = params[5]
    
    # 1. Ultimate Strategy (Simplified VIX Filter)
    vix_sma = vix.rolling(vix_ma_win).mean()
    vix_sig = pd.Series(0, index=prices.index)
    vix_sig[vix < vix_sma] = 1 # Calm
    vix_sig[vix > vix_sma] = -1 # Fear
    
    w_ult_ts = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Vectorized Approx for speed in GA
    # Trad
    # w_ult_ts['SPY'] = np.where(vix_sig > 0, 0.45, np.where(vix_sig < 0, 0.15, 0.30))
    # w_ult_ts['TLT'] = np.where(vix_sig > 0, 0.10, np.where(vix_sig < 0, 0.35, 0.22))
    
    # Faster Vectorization
    mask_bull = (vix_sig > 0)
    mask_bear = (vix_sig < 0)
    
    if 'SPY' in prices.columns:
        w_ult_ts.loc[mask_bull, 'SPY'] = 0.45
        w_ult_ts.loc[mask_bear, 'SPY'] = 0.15
        w_ult_ts.loc[(~mask_bull) & (~mask_bear), 'SPY'] = 0.30
        
    if 'TLT' in prices.columns:
        w_ult_ts.loc[mask_bull, 'TLT'] = 0.10
        w_ult_ts.loc[mask_bear, 'TLT'] = 0.35
        w_ult_ts.loc[(~mask_bull) & (~mask_bear), 'TLT'] = 0.22
        
    if 'BTC-USD' in prices.columns:
        # Simple mom check
        btc_ret = prices['BTC-USD'].pct_change()
        # Cannot easily vectorize mom check inside backtest loop without lookahead or slow loop
        # Simplify: Fixed 20%
        w_ult_ts['BTC-USD'] = 0.20
        
    r_ult = (w_ult_ts.shift(1) * prices.pct_change().fillna(0)).sum(axis=1)
    
    # 2. HRP Proxy (Inv Vol)
    rets = prices.pct_change().fillna(0)
    vol = rets.rolling(hrp_look).std()
    vol = vol.replace(0, 0.01)
    inv_vol = 1/vol
    w_hrp_ts = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    r_hrp = (w_hrp_ts.shift(1) * rets).sum(axis=1)
    
    # 3. UUP Trend
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        sma_f = uup.rolling(uup_fast).mean()
        sma_s = uup.rolling(uup_slow).mean()
        uup_sig = (sma_f > sma_s).astype(float)
        r_uup_strat = (uup_sig.shift(1) * prices['UUP'].pct_change().fillna(0))
    else:
        uup_sig = pd.Series(0, index=prices.index)
        r_uup_strat = pd.Series(0, index=prices.index)
        
    # 4. Combine
    # UUP takes 'uup_size' off the top if active
    # Remaining '1 - uup_size' is split by 'alloc_split'
    
    # If UUP Active:
    # Alloc UUP = uup_size
    # Alloc Rest = 1 - uup_size
    # Else:
    # Alloc UUP = 0
    # Alloc Rest = 1.0
    
    alloc_uup = uup_sig.shift(1) * uup_size
    alloc_rest = 1.0 - alloc_uup
    
    alloc_h = alloc_rest * alloc_split
    alloc_u = alloc_rest * (1 - alloc_split)
    
    port_ret = alloc_u * r_ult + alloc_h * r_hrp + alloc_uup * r_uup_strat
    return port_ret

def get_fitness(port_ret):
    if len(port_ret) < 100: return -999
    ann = port_ret.mean() * 252
    vol = port_ret.std() * np.sqrt(252)
    if vol == 0: return -999
    sharpe = ann / vol
    return sharpe

# =============================================================================
# 3. GENETIC ALGORITHM
# =============================================================================

def run_evolution(prices, vix):
    # Split
    split_date = '2020-01-01'
    idx_train = prices.index[prices.index < split_date]
    idx_test = prices.index[prices.index >= split_date]
    
    prices_train = prices.loc[idx_train]
    vix_train = vix.loc[idx_train]
    prices_test = prices.loc[idx_test]
    vix_test = vix.loc[idx_test]
    
    print("\n🧬 STARTING EVOLUTION...")
    print(f"   Train: {len(idx_train)} days | Test: {len(idx_test)} days")
    
    # Pop Size
    POP_SIZE = 50
    GENS = 5
    
    # Init Pop (Random)
    # 0: vix_ma (10-60)
    # 1: hrp_look (20-252)
    # 2: uup_fast (20-100)
    # 3: uup_slow (120-300)
    # 4: alloc_split (0.0-1.0) -> HRP skew
    # 5: uup_size (0.0-0.4)
    
    generation_scores = []
    
    population = []
    for _ in range(POP_SIZE):
        ind = [
            random.randint(10, 60),
            random.randint(20, 252),
            random.randint(20, 100),
            random.randint(120, 300),
            random.uniform(0.3, 0.7),
            random.uniform(0.0, 0.4)
        ]
        population.append(ind)
        
    # Baseline (Heuristic)
    base_params = [20, 126, 50, 200, 0.5, 0.2]
    r_base_train = run_system_logic(prices_train, vix_train, base_params)
    s_base_train = get_fitness(r_base_train)
    r_base_test = run_system_logic(prices_test, vix_test, base_params)
    s_base_test = get_fitness(r_base_test)
    
    print(f"   Baseline (Heuristic) Train Sharpe: {s_base_train:.2f}")
    
    # Evolution Loop
    best_ind = None
    best_fit = -999
    
    for gen in range(GENS):
        # Evaluate
        fitnesses = []
        for ind in population:
            r = run_system_logic(prices_train, vix_train, ind)
            f = get_fitness(r)
            fitnesses.append((ind, f))
            
            if f > best_fit:
                best_fit = f
                best_ind = ind
        
        # Sort
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        top_half = fitnesses[:POP_SIZE//2]
        
        # Report
        avg_fit = sum(f for i, f in fitnesses) / len(fitnesses)
        print(f"   Gen {gen+1}: Best {fitnesses[0][1]:.2f} | Avg {avg_fit:.2f}")
        
        # Next Gen (Selection + Crossover + Mutation)
        new_pop = [ind for ind, fit in top_half] # Elitism (Top 50% survive)
        
        while len(new_pop) < POP_SIZE:
            # Select 2 parents
            p1 = random.choice(top_half)[0]
            p2 = random.choice(top_half)[0]
            
            # Crossover
            child = []
            for k in range(len(p1)):
                child.append(p1[k] if random.random() < 0.5 else p2[k])
                
            # Mutation (10% chance)
            if random.random() < 0.2:
                # Mutate random gene
                gene_idx = random.randint(0, 5)
                if gene_idx == 0: child[0] = random.randint(10, 60)
                elif gene_idx == 1: child[1] = random.randint(20, 252)
                elif gene_idx == 2: child[2] = random.randint(20, 100)
                elif gene_idx == 3: child[3] = random.randint(120, 300)
                elif gene_idx == 4: child[4] = random.uniform(0.3, 0.7)
                elif gene_idx == 5: child[5] = random.uniform(0.0, 0.4)
                
            new_pop.append(child)
            
        population = new_pop
        
    print("\n🏆 EVOLUTION COMPLETE")
    print(f"   Best Genes: {best_ind}")
    print(f"   [VixMA, HRP, UUP_F, UUP_S, Split, UUP_Sz]")
    
    # Test on OOS
    r_oos = run_system_logic(prices_test, vix_test, best_ind)
    s_oos = get_fitness(r_oos)
    
    print("-" * 60)
    print(f"{'Metric':<20} {'Baseline':<10} {'Evolved':<10} {'Diff'}")
    print("-" * 60)
    print(f"{'Train Sharpe':<20} {s_base_train:<10.2f} {best_fit:<10.2f} {best_fit - s_base_train:+.2f}")
    print(f"{'OOS Sharpe':<20} {s_base_test:<10.2f} {s_oos:<10.2f} {s_oos - s_base_test:+.2f}")
    print("-" * 60)
    
    if s_oos > s_base_test + 0.1:
        print("✅ SUCCESS: Evolved parameters generalized OOS!")
    elif s_oos > s_base_test:
        print("⚠️ MARGINAL: Slight improvement OOS.")
    else:
        print("❌ OVERFIT: Evolved params failed OOS. Baseline is robust.")

if __name__ == "__main__":
    prices, vix = fetch_data()
    run_evolution(prices, vix)
