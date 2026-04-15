"""
Evolutionary Alpha Hunt (Genetic Algorithm)
============================================

Breeding the "Ultimate Strategy" to find the optimal parameter set.
Genome: [SMA_Window, RSI_Window, RSI_Buy, RSI_Sell, Vol_Target]

Process:
1. Initialize Population (Random Genomes)
2. Evaluate Fitness (Sharpe Ratio on Backtest)
3. Selection (Tournament)
4. Crossover (Mix Parents)
5. Mutation (Random Variance)
6. Repeat for N Generations.

RUN: python evolutionary_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. THE GENOME
# =============================================================================

class Genome:
    def __init__(self, params=None):
        if params:
            self.params = params
        else:
            # Random Initialization
            self.params = {
                'sma_window': random.randint(20, 300),
                'rsi_window': random.randint(5, 50),
                'rsi_buy': random.randint(10, 50),
                'rsi_sell': random.randint(50, 90),
                'vix_thresh': random.randint(15, 35),
                'crypto_weight': random.uniform(0.1, 0.6)
            }
        self.fitness = 0.0
        self.sharpe = 0.0
        self.ret = 0.0
        
    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            key = random.choice(list(self.params.keys()))
            if key == 'sma_window': self.params[key] = random.randint(20, 300)
            elif key == 'rsi_window': self.params[key] = random.randint(5, 50)
            elif key == 'rsi_buy': self.params[key] = random.randint(10, 50)
            elif key == 'rsi_sell': self.params[key] = random.randint(50, 90)
            elif key == 'vix_thresh': self.params[key] = random.randint(15, 35)
            elif key == 'crypto_weight': self.params[key] = random.uniform(0.1, 0.6)

# =============================================================================
# 2. FITNESS FUNCTION (Backtest)
# =============================================================================

def fetch_evo_data():
    print("🧬 Fetching Evolutionary Dataset...")
    tickers = ['SPY', 'BTC-USD', '^VIX'] # Basic components
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                 prices = data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    # Standardize VIX (Ticker might be ^VIX)
    # yfinance sometimes maps ^VIX to something else.
    # Let's check col names
    # Usually it's '^VIX' or 'VIX'
    
    # Fill NAs
    prices = prices.ffill().dropna()
    return prices

def evaluate_genome(genome, prices):
    """
    Backtest the genome parameters.
    Strategy: 
      - If SPY > SMA(window) AND VIX < thresh:
          - If RSI < buy: Allocation = Crypto_Weight * 1.5 (Aggressive)
          - Else: Allocation = Crypto_Weight (Base)
      - Else (Bear Regime):
          - Cash / Defensive
    
    Simplified Logic for Speed:
      - Regime = SPY > SMA
      - If Regime: Buy Mix (SPY + Crypto)
      - Else: Cash
    """
    p = genome.params
    
    # Extract Series
    if 'SPY' not in prices.columns or 'BTC-USD' not in prices.columns:
        return 0 # Fail
        
    spy = prices['SPY']
    btc = prices['BTC-USD']
    if '^VIX' in prices.columns:
        vix = prices['^VIX']
    else:
        vix = pd.Series(20, index=prices.index) # Fallback
        
    # Indicators
    sma = spy.rolling(p['sma_window']).mean()
    
    # RSI (Vectorized approx)
    delta = spy.diff()
    gain = (delta.where(delta > 0, 0)).rolling(p['rsi_window']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(p['rsi_window']).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Signals
    regime = (spy > sma) & (vix < p['vix_thresh'])
    
    # Allocation Logic
    # If Regime:
    #   Weight = Crypto_Weight in BTC, (1-Crypto) in SPY
    #   But RSI modifier:
    #     If RSI < Buy (Oversold): Leverage BTC? Or just 100% BTC?
    #     Let's essentially say: Normal = p['crypto_weight']
    #     Oversold = p['crypto_weight'] * 1.5 (capped at 1.0)
    
    # Vectorized Weight calculation
    w_btc = pd.Series(0.0, index=spy.index)
    w_spy = pd.Series(0.0, index=spy.index)
    
    # Base Bull
    w_btc[regime] = p['crypto_weight']
    w_spy[regime] = 1.0 - p['crypto_weight']
    
    # RSI Boost (Oversold Dip Buy)
    dip = regime & (rsi < p['rsi_buy'])
    w_btc[dip] = min(p['crypto_weight'] * 1.5, 1.0)
    w_spy[dip] = 1.0 - w_btc[dip]
    
    # RSI Cut (Overbought Trim)
    peak = regime & (rsi > p['rsi_sell'])
    w_btc[peak] = p['crypto_weight'] * 0.5
    w_spy[peak] = 1.0 - w_btc[peak]
    
    # Returns
    port_ret = (w_btc.shift(1) * btc.pct_change()) + (w_spy.shift(1) * spy.pct_change())
    port_ret = port_ret.dropna()
    
    # Metrics
    if len(port_ret) < 100 or port_ret.std() == 0:
        return 0, 0
        
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    cum_ret = (1 + port_ret).cumprod().iloc[-1] - 1
    
    return sharpe, cum_ret

# =============================================================================
# 3. GENETIC ENGINE
# =============================================================================

def crossover(parent1, parent2):
    # Uniform Crossover
    child_params = {}
    for key in parent1.params:
        if random.random() < 0.5:
            child_params[key] = parent1.params[key]
        else:
            child_params[key] = parent2.params[key]
    return Genome(child_params)

def run_evolution():
    prices = fetch_evo_data()
    
    POP_SIZE = 50
    GENERATIONS = 10
    
    # Genesis
    print(f"🐣 Initializing Population ({POP_SIZE})...")
    population = [Genome() for _ in range(POP_SIZE)]
    
    best_ever = None
    
    print(f"{'Gen':<5} {'Best Sharpe':<12} {'Return':<10} {'Params'}")
    print("-" * 80)
    
    for gen in range(GENERATIONS):
        # Evaluate
        for indiv in population:
            s, r = evaluate_genome(indiv, prices)
            indiv.fitness = s # Objective: Maximize Sharpe
            indiv.sharpe = s
            indiv.ret = r
            
            if best_ever is None or s > best_ever.sharpe:
                best_ever = indiv
        
        # Sort
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_gen = population[0]
        
        # Print
        metrics = f"{best_gen.sharpe:.2f}"
        ret_s = f"{best_gen.ret:.1%}"
        params_s = str(best_gen.params)
        print(f"{gen+1:<5} {metrics:<12} {ret_s:<10} {params_s}")
        
        # Selection (Top 50%)
        survivors = population[:POP_SIZE//2]
        
        # Breeding
        next_gen = survivors[:] # Elitism
        while len(next_gen) < POP_SIZE:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover(p1, p2)
            child.mutate(0.2) 
            next_gen.append(child)
            
        population = next_gen
        
    print("\n🏆 EVOLUTION COMPLETE")
    print(f"Top Sharpe: {best_ever.sharpe:.2f}")
    print(f"Top Params: {best_ever.params}")
    
    if best_ever.sharpe > 1.42 + 0.1:
        print("✅ SUCCESS: Evolution beat the human design!")
    else:
        print("❌ CONVERGED: Human design was already near-optimal.")

if __name__ == "__main__":
    print("="*60)
    print("🧬 EVOLUTIONARY ALPHA HUNT (GENETIC ALGORITHM)")
    print("="*60)
    run_evolution()
    print("\n" + "=" * 60)
