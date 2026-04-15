
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def optimize_portfolio(rets):
    """Find weights that maximize Sharpe Ratio."""
    def neg_sharpe(weights, rets):
        p_ret = (rets * weights).sum(axis=1)
        mean = p_ret.mean() * 252
        vol = p_ret.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0
        return -sharpe

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(rets.shape[1]))
    init_guess = [1.0 / rets.shape[1]] * rets.shape[1]
    
    # Check if we have enough data
    if len(rets) < 20:
        return init_guess
        
    try:
        opt = minimize(neg_sharpe, init_guess, args=(rets,), method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x
    except:
        return init_guess

def run_calculation():
    print("Fetching data...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', 'UUP', # Trad
        'VUG', 'VTV', 'RSP', # Factors/Breadth
        '^FVX', '^TYX', '^VIX', # Macro
        'BTC-USD' # Crypto
    ]
    data = yf.download(tickers, start='2015-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    prices = prices.ffill().dropna()
    
    # returns
    rets = prices.pct_change().fillna(0)
    
    # 2. Ultimate
    vix = prices['^VIX']
    vix_ma = vix.rolling(20).mean()
    signal = pd.Series(0, index=prices.index)
    signal[vix < vix_ma] = 1 # Calm
    signal[vix > vix_ma] = -1 # Fear
    
    w_ult = pd.DataFrame(0.0, index=prices.index, columns=['SPY', 'TLT', 'BTC-USD'])
    mask_bull = (signal > 0)
    mask_bear = (signal < 0)
    
    w_ult.loc[mask_bull, 'SPY'] = 0.45; w_ult.loc[mask_bull, 'TLT'] = 0.10
    w_ult.loc[mask_bear, 'SPY'] = 0.15; w_ult.loc[mask_bear, 'TLT'] = 0.35
    w_ult.loc[(~mask_bull) & (~mask_bear), 'SPY'] = 0.30; w_ult.loc[(~mask_bull) & (~mask_bear), 'TLT'] = 0.22
    w_ult['BTC-USD'] = 0.20
    
    sp = rets['SPY'] if 'SPY' in rets.columns else 0
    tl = rets['TLT'] if 'TLT' in rets.columns else 0
    bt = rets['BTC-USD'] if 'BTC-USD' in rets.columns else 0
    
    r_ult = (w_ult['SPY'].shift(1)*sp + w_ult['TLT'].shift(1)*tl + w_ult['BTC-USD'].shift(1)*bt)
    
    # 3. HRP
    cols_hrp = [c for c in ['SPY', 'TLT', 'GLD', 'IEF'] if c in rets.columns]
    vol = rets[cols_hrp].rolling(126).std()
    inv_vol = 1 / vol.replace(0, 0.01)
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * rets[cols_hrp]).sum(axis=1)
    
    # 4. Dollar Trend
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        uup_fast = uup.rolling(50).mean()
        uup_slow = uup.rolling(200).mean()
        uup_sig = (uup_fast > uup_slow).astype(float)
        r_uup = (uup_sig.shift(1) * rets['UUP'])
    else:
        r_uup = pd.Series(0, index=prices.index)
    
    # Optimization
    strat_rets = pd.DataFrame({
        'Ultimate': r_ult,
        'HRP': r_hrp,
        'Dollar Trend': r_uup
    }).dropna()
    
    print("Optimizing...")
    opt_w = optimize_portfolio(strat_rets)
    
    print("\n" + "="*40)
    print("SCIENTIFICALLY OPTIMAL WEIGHTS (Sharpe)")
    print("="*40)
    print(f"Ultimate (Growth):   {opt_w[0]:.1%}")
    print(f"HRP (Safety):        {opt_w[1]:.1%}")
    print(f"Dollar Trend (Hedge):{opt_w[2]:.1%}")
    print("="*40)

    # Performance Comparison
    # Reconstruct returns with optimal weights
    r_tri = (opt_w[0] * r_ult + opt_w[1] * r_hrp + opt_w[2] * r_uup)
    
    # Calculate Metrics
    def calc_metrics(r):
        cum = (1 + r).prod() - 1
        n_years = len(r) / 252
        cagr = (1 + r).prod()**(1/n_years) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = r.mean() * 252 / vol if vol > 0 else 0
        
        # Max Drawdown
        wealth = (1 + r).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth - peak) / peak
        max_dd = drawdown.min()
        
        return cum, cagr, sharpe, max_dd

    spy_cum, spy_cagr, spy_sharpe, spy_dd = calc_metrics(rets['SPY'])
    tri_cum, tri_cagr, tri_sharpe, tri_dd = calc_metrics(r_tri)
    
    ult_cum, ult_cagr, ult_sharpe, ult_dd = calc_metrics(r_ult)
    hrp_cum, hrp_cagr, hrp_sharpe, hrp_dd = calc_metrics(r_hrp)
    uup_cum, uup_cagr, uup_sharpe, uup_dd = calc_metrics(r_uup)

    print("\n" + "="*80)
    print(f"{'METRIC':<15} {'S&P 500':<10} {'TRIFECTA':<10} | {'ULTIMATE':<10} {'HRP':<10} {'$ TREND':<10}")
    print("="*80)
    print(f"{'Total Return':<15} {spy_cum:>9.1%} {tri_cum:>9.1%} | {ult_cum:>9.1%} {hrp_cum:>9.1%} {uup_cum:>9.1%}")
    print(f"{'CAGR':<15} {spy_cagr:>9.1%} {tri_cagr:>9.1%} | {ult_cagr:>9.1%} {hrp_cagr:>9.1%} {uup_cagr:>9.1%}")
    print(f"{'Sharpe Ratio':<15} {spy_sharpe:>9.2f} {tri_sharpe:>9.2f} | {ult_sharpe:>9.2f} {hrp_sharpe:>9.2f} {uup_sharpe:>9.2f}")
    print(f"{'Max Drawdown':<15} {spy_dd:>9.1%} {tri_dd:>9.1%} | {ult_dd:>9.1%} {hrp_dd:>9.1%} {uup_dd:>9.1%}")
    print("="*80)

    if tri_cum > spy_cum:
        print(f"\n✅ YES. It beats the S&P by {tri_cum - spy_cum:.1%}.")
    else:
        print(f"\n❌ NO. It trails the S&P by {spy_cum - tri_cum:.1%}.")

if __name__ == "__main__":
    run_calculation()
