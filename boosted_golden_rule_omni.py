
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 1. WEIRD DATA (The "Omni" Signal)
# =================================
WEIRD_DATA = {
    "netflix": {
        2000: 10.0, 2001: 10.0, 2002: 10.0, 2003: 10.0, 2004: 10.0,
        2005: 10.0, 2006: 10.0, 2007: 10.0, 2008: 10.0, 2009: 10.0, # Interpolated/Flat history logic
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0,
    },
    "cheese": {
        2000: 30.0, 2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5,
    },
    "coffee": {
        2000: 3.5, 2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32,
    },
}

def get_erp_inflation_signal(date):
    """Returns True if 'Weird Data' implies inflation (Cost Push)."""
    year = date.year
    # Simple logic: If Cheese & Coffee are rising > 5%, it's inflation.
    yoy_cheese = 0
    yoy_coffee = 0
    
    if year in WEIRD_DATA['cheese'] and year-1 in WEIRD_DATA['cheese']:
        yoy_cheese = (WEIRD_DATA['cheese'][year] / WEIRD_DATA['cheese'][year-1]) - 1
        
    if year in WEIRD_DATA['coffee'] and year-1 in WEIRD_DATA['coffee']:
        yoy_coffee = (WEIRD_DATA['coffee'][year] / WEIRD_DATA['coffee'][year-1]) - 1
        
    # Inflation Signal: Food costs rising fast
    loss_of_purchasing_power = (yoy_cheese > 0.03) or (yoy_coffee > 0.05)
    return loss_of_purchasing_power

# 2. DATA
# =======
def fetch_data():
    print("Fetching Omni Data...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'UUP', 'XLE', # Trad + Commods
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', # Crypto
        '^VIX'
    ]
    data = yf.download(tickers, period='max', interval='1d', progress=False)
    try:
        prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    except:
        prices = data['Close']
    
    # Filter to 2004+
    return prices[prices.index >= '2004-01-01'].ffill()

# 3. STRATEGY ENGINE
# ==================
def calc_strategy(prices):
    rets = prices.pct_change().fillna(0)
    
    # --- A. Smart Crypto (Altseason + Vol Control) ---
    btc = prices.get('BTC-USD', pd.Series(0, index=prices.index))
    btc_vol = btc.pct_change().rolling(30).std() * np.sqrt(365) * 100
    risk_off_crypto = (btc_vol > 100).shift(1).fillna(False)
    
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    btc_mom = btc.pct_change(14)
    if avail_alts:
        alt_index = prices[avail_alts].mean(axis=1)
        alt_mom = alt_index.pct_change(14)
    else:
        alt_mom = pd.Series(-999, index=prices.index)
        
    altseason = (alt_mom > btc_mom).shift(1).fillna(False)
    
    r_crypto_smart = pd.Series(0.0, index=rets.index)
    trade_days = ~risk_off_crypto
    
    if avail_alts:
        mask_alts = trade_days & altseason
        r_crypto_smart[mask_alts] = rets[avail_alts].mean(axis=1)[mask_alts]
        
        mask_btc = trade_days & ~altseason
        r_crypto_smart[mask_btc] = rets['BTC-USD'][mask_btc]
    else:
        r_crypto_smart[trade_days] = rets.get('BTC-USD', 0)[trade_days]

    # --- B. Ultimate (Enhanced) ---
    vix = prices['^VIX'].fillna(20)
    bull_vix = (vix < vix.rolling(20).mean()).shift(1).fillna(False)
    
    r_ult = pd.Series(0.0, index=rets.index)
    
    # Weights for Ultimate
    # Bull VIX: 45 SPY / 10 TLT / 5 GLD / 40 Crypto
    # Bear VIX: 15 SPY / 35 TLT / 10 GLD / 40 Crypto
    
    # Bull
    r_trad_bull = 0.45*rets['SPY'] + 0.10*rets['TLT'] + 0.05*rets.get('GLD', 0)
    r_ult[bull_vix] = r_trad_bull[bull_vix] + 0.40*r_crypto_smart[bull_vix]
    
    # Bear
    r_trad_bear = 0.15*rets['SPY'] + 0.35*rets['TLT'] + 0.10*rets.get('GLD', 0)
    r_ult[~bull_vix] = r_trad_bear[~bull_vix] + 0.40*r_crypto_smart[~bull_vix]

    # --- C. HRP (Safety) ---
    hrp_assets = ['SPY', 'TLT', 'GLD']
    avail_hrp = [c for c in hrp_assets if c in rets.columns]
    vol_hrp = rets[avail_hrp].rolling(20).std()
    inv_vol = 1 / vol_hrp.replace(0, np.nan)
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * rets[avail_hrp]).sum(axis=1)

    # --- D. ERP Overlay (Inflation Regime) ---
    # Logic: If Inflation Regime (Cheese/Coffee Up), pivot HRP/Safety towards Real Assets (GLD/XLE).
    # We modify r_hrp dynamically.
    
    # Calculate Inflation Signal mask
    inflation_mask = pd.Series(False, index=rets.index)
    dates = rets.index
    for i in range(len(dates)):
        inflation_mask.iloc[i] = get_erp_inflation_signal(dates[i])
        
    inflation_mask = inflation_mask.shift(1).fillna(False) # Lag by 1 day to be safe (actually annual but safe)
    
    # Create "Real Asset HRP" for Inflation Regime
    # SPY, XLE, GLD (No Bonds)
    real_assets = ['SPY', 'XLE', 'GLD']
    avail_real = [c for c in real_assets if c in rets.columns]
    vol_real = rets[avail_real].rolling(20).std()
    inv_vol_real = 1 / vol_real.replace(0, np.nan)
    w_real = inv_vol_real.div(inv_vol_real.sum(axis=1), axis=0).shift(1).fillna(0)
    r_real_hrp = (w_real * rets[avail_real]).sum(axis=1)
    
    # E. The OMNI Switch (Golden Rule)
    # ================================
    spy = prices['SPY']
    ma_short = spy.rolling(150).mean() # Slightly faster than 200 for Omni
    ma_long = spy.rolling(200).mean()
    gold_rule_bull = (spy > ma_long).shift(1).fillna(False)
    
    r_omni = pd.Series(0.0, index=rets.index)
    
    mask_bull = gold_rule_bull
    mask_bear = ~gold_rule_bull
    
    # 1. Bull Market -> Ultimate (Growth + Crypto)
    # UNLESS Inflation is insane? No, in Bull market we trust trend.
    r_omni[mask_bull] = r_ult[mask_bull]
    
    # 2. Bear Market -> Safety
    # Standard: HRP (Bonds/Gold/Spy)
    # Inflationary (ERP): Real HRP (XLE/Gold/Spy) - NO BONDS
    
    mask_bear_normal = mask_bear & ~inflation_mask
    mask_bear_inflation = mask_bear & inflation_mask
    
    r_omni[mask_bear_normal] = r_hrp[mask_bear_normal]
    r_omni[mask_bear_inflation] = r_real_hrp[mask_bear_inflation]
    
    
    # ANALYSIS
    # Compare Omni vs Previous
    
    print("\nGOLDEN OMNI STRATEGY (Boosted + Crypto + ERP Inflation)")
    print("="*65)
    print(f"{'Strategy':<30} {'Sharpe':<8} {'Return':<10} {'MaxDD':<10}")
    print("-" * 65)
    
    compare = {
        'Golden Omni': r_omni,
        'Crypto-Enhanced Boost': r_ult, # Approx the bull leg
        'SPY Buy & Hold': rets['SPY']
    }
    
    res = []
    for name, r in compare.items():
        cum = (1 + r).prod() - 1
        vol = r.std() * np.sqrt(252)
        sharpe = r.mean() * 252 / vol if vol > 0 else 0
        w = (1 + r).cumprod()
        dd = (w / w.cummax()) - 1
        mdd = dd.min()
        res.append((name, sharpe, cum, mdd))
        
    res.sort(key=lambda x: x[1], reverse=True)
    for n,s,c,d in res:
        print(f"{n:<30} {s:<8.2f} {c:<10.0%} {d:<10.1%}")
    print("=" * 65)

if __name__ == "__main__":
    p = fetch_data()
    calc_strategy(p)
