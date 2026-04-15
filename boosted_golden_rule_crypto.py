
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    print("Fetching Data with Extended Crypto Universe...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'UUP', # Trad
        'BTC-USD', 'ETH-USD', 'DOGE-USD', 'SOL-USD', # Crypto (Robinhood avail)
        '^VIX'
    ]
    data = yf.download(tickers, period='max', interval='1d', progress=False)
    
    try:
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    except:
        adj_close = data['Close']
        
    return adj_close[adj_close.index >= '2004-01-01'].ffill()

def calc_strategy_returns(prices):
    rets = prices.pct_change().fillna(0)
    
    # 1. CRYPTO COMMANDER LOGIC
    # =========================
    # Strategy: 
    # if BTC Vol > 100 -> Cash (0% Crypto)
    # elif Alt Mom > BTC Mom -> 100% Alts (Aggressive)
    # else -> 100% BTC (Dominance)
    
    btc = prices['BTC-USD'] if 'BTC-USD' in prices else pd.Series(0, index=prices.index)
    
    # A. Volatility Check (Risk Management)
    btc_vol = btc.pct_change().rolling(30).std() * np.sqrt(365) * 100
    risk_off_crypto = (btc_vol > 100).shift(1).fillna(False)
    
    # B. Altseason Signal (Momentum)
    # Basket of Alts
    alts = ['ETH-USD', 'DOGE-USD', 'SOL-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    # Simple Mom(14d)
    btc_mom = btc.pct_change(14)
    if avail_alts:
        alt_basket_price = prices[avail_alts].mean(axis=1) # Simplified index
        alt_mom = alt_basket_price.pct_change(14)
    else:
        alt_mom = pd.Series(-999, index=prices.index) # Default to BTC if no data
        
    altseason = (alt_mom > btc_mom).shift(1).fillna(False)
    
    # C. Calculate "Smart Crypto" Return
    # Construct the Crypto Portfolio Return stream
    r_crypto_smart = pd.Series(0.0, index=rets.index)
    
    # If Risk ON (Vol < 100):
    #   If Altseason: Return = Avg(Alts)
    #   Else: Return = BTC
    # If Risk OFF: Return = 0 (Cash)
    
    # Using boolean logic
    # Mask: Valid Trading Days (Not Risk Off)
    trade_days = ~risk_off_crypto
    
    # On Trade Days, check rotation
    # Note: If alt data missing, fallback to BTC
    
    if avail_alts:
        r_alts = rets[avail_alts].mean(axis=1)
        # Apply Alt returns where Altseason is True AND Trading Day is True
        mask_alts = trade_days & altseason
        r_crypto_smart[mask_alts] = r_alts[mask_alts]
        
        # Apply BTC returns where Altseason is False AND Trading Day is True
        mask_btc = trade_days & ~altseason
        r_crypto_smart[mask_btc] = rets['BTC-USD'][mask_btc]
    else:
        # No alts, just BTC controlled by vol
        r_crypto_smart[trade_days] = rets['BTC-USD'][trade_days]
        
    
    # 2. ULTIMATE STRATEGY (UPDATED)
    # ==============================
    # Now uses "Smart Crypto" component instead of raw BTC
    
    vix = prices['^VIX'].fillna(20)
    vix_ma = vix.rolling(20).mean()
    bull_vix = vix < vix_ma
    
    # Weights driven by VIX
    # Bull VIX: 45% SPY, 10% TLT, 5% GLD, 40% Smart Crypto
    # Bear VIX: 15% SPY, 35% TLT, 10% GLD, 40% Smart Crypto
    # Note: The user likes high returns. 
    # Let's keep the structure but swap BTC-USD col for r_crypto_smart series
    
    r_ult_enhanced = pd.Series(0.0, index=rets.index)
    
    # Bull VIX Contribution
    # 60% Trad + 40% Crypto
    r_trad_bull = (0.45 * rets['SPY']) + (0.10 * rets['TLT']) + (0.05 * rets.get('GLD', 0))
    r_ult_enhanced[bull_vix] = r_trad_bull[bull_vix] + (0.40 * r_crypto_smart[bull_vix])
    
    # Bear VIX Contribution
    # 60% Trad + 40% Crypto
    r_trad_bear = (0.15 * rets['SPY']) + (0.35 * rets['TLT']) + (0.10 * rets.get('GLD', 0))
    r_ult_enhanced[~bull_vix] = r_trad_bear[~bull_vix] + (0.40 * r_crypto_smart[~bull_vix])
    
    
    # 3. HRP STRATEGY (Unchanged Safety)
    # ==================================
    hrp_assets = ['SPY', 'TLT', 'GLD']
    avail_hrp = [c for c in hrp_assets if c in rets.columns]
    vol = rets[avail_hrp].rolling(20).std()
    inv_vol = 1 / vol.replace(0, np.nan)
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * rets[avail_hrp]).sum(axis=1)


    # 4. MASTER SWITCH (Golden Rule)
    # ==============================
    spy = prices['SPY']
    ma200 = spy.rolling(200).mean()
    regime_bull = (spy > ma200).shift(1).fillna(False)
    
    # 5. NEW BOOSTED STRATEGY
    # =======================
    # Bull Market: Ultimate Enhanced (with Smart Crypto)
    # Bear Market: HRP
    
    r_final = pd.Series(0.0, index=rets.index)
    r_final[regime_bull] = r_ult_enhanced[regime_bull]
    r_final[~regime_bull] = r_hrp[~regime_bull]
    
    
    # ANALYSIS
    print("\nCRYPTO-ENHANCED BOOSTED STRATEGY")
    print("Logic: Golden Rule -> Ultimate (w/ Altseason & Vol Control) / HRP")
    print("=" * 65)
    print(f"{'Strategy':<30} {'Sharpe':<8} {'Return':<10} {'MaxDD':<10}")
    print("-" * 65)
    
    compare = {
        'Crypto-Enhanced Boost': r_final,
        'Previous Boost (Static BTC)': calc_previous_boost(prices, regime_bull, r_hrp),
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

def calc_previous_boost(prices, regime_bull, r_hrp):
    # Quick re-calc of previous best for comparison
    # Ultimate Static
    rets = prices.pct_change().fillna(0)
    vix = prices['^VIX'].fillna(20)
    bull_vix = vix < vix.rolling(20).mean()
    
    r_ult = pd.Series(0.0, index=rets.index)
    # Simple static mix used previously approx
    # 45 SPY / 10 TLT / 20 BTC
    # vs Bear: 15 SPY / 35 TLT / 20 BTC
    btc = rets.get('BTC-USD', 0)
    
    r_bull = (0.45 * rets['SPY']) + (0.10 * rets['TLT']) + (0.20 * btc)
    r_bear = (0.15 * rets['SPY']) + (0.35 * rets['TLT']) + (0.20 * btc)
    
    r_ult[bull_vix] = r_bull[bull_vix]
    r_ult[~bull_vix] = r_bear[~bull_vix]
    
    r_combined = pd.Series(0.0, index=rets.index)
    r_combined[regime_bull] = r_ult[regime_bull]
    r_combined[~regime_bull] = r_hrp[~regime_bull]
    
    return r_combined

if __name__ == "__main__":
    p = fetch_data()
    calc_strategy_returns(p)
