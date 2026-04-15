"""
Macro-Seasonality Alpha Hunt
============================

Testing Calendar Anomalies.
Hypothesis: Institutional flows and behavioral biases create predictable seasonal patterns.

Strategies:
1. Sell in May (Halloween Indicator): Long SPY Nov-Apr, Long TLT May-Oct.
2. Santa Claus Rally: Long Last 5 days Dec + First 2 days Jan.
3. Pre-Holiday Drift: Long Day before Market Holidays.

RUN: python seasonal_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. FETCH DATA
# =============================================================================

def fetch_data():
    print("📅 Fetching Seasonal Data (SPY, TLT)...")
    tickers = ['SPY', 'TLT']
    data = yf.download(tickers, start='2000-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days")
    return prices

# =============================================================================
# 2. STRATEGIES
# =============================================================================

def strategy_sell_in_may(prices):
    """
    Long SPY in Winter (Nov-Apr), Long TLT in Summer (May-Oct).
    """
    if 'SPY' not in prices.columns or 'TLT' not in prices.columns: return None
    
    spy = prices['SPY'].pct_change()
    tlt = prices['TLT'].pct_change()
    
    # Signal
    months = prices.index.month
    # Winter: 11, 12, 1, 2, 3, 4
    is_winter = (months >= 11) | (months <= 4)
    
    weights_spy = pd.Series(0.0, index=prices.index)
    weights_tlt = pd.Series(0.0, index=prices.index)
    
    weights_spy[is_winter] = 1.0
    weights_tlt[~is_winter] = 1.0 # Summer
    
    # Lag 1 day? No, this is monthly logic, usually executed at close of prev month?
    # Let's assume we hold correctly for the day.
    # If today is Nov 1, is_winter=True. Return is (Nov1/Oct31)-1.
    # So we need to have bought Oct 31 Close.
    # So Signal should be shifted?
    # If we use simple boolean masking on returns, it implies we were invested ON that day.
    # Which corresponds to buying the Close of T-1.
    
    strat_ret = (weights_spy * spy + weights_tlt * tlt).fillna(0)
    return strat_ret

def strategy_santa_rally(prices):
    """
    Long SPY Last 5 days of Dec, First 2 days of Jan. Cash otherwise.
    """
    if 'SPY' not in prices.columns: return None
    
    spy = prices['SPY'].pct_change()
    signal = pd.Series(0.0, index=prices.index)
    
    # Group by Year to find last days
    # This is tricky without iteration or complex grouping
    
    # Vectorized approach:
    # 1. Get Day of Year? No, leap years.
    # 2. Iterate years (fast enough for 20 years)
    
    years = prices.index.year.unique()
    
    for y in years:
        # Get Dec dates
        dec_dates = prices[prices.index.year == y].index
        dec_dates = dec_dates[dec_dates.month == 12]
        
        # Get Jan dates of next year
        jan_dates = prices[prices.index.year == y+1].index
        jan_dates = jan_dates[jan_dates.month == 1]
        
        if len(dec_dates) >= 5:
            last_5 = dec_dates[-5:]
            signal[last_5] = 1.0
            
        if len(jan_dates) >= 2:
            first_2 = jan_dates[:2]
            signal[first_2] = 1.0
            
    return (signal * spy).fillna(0)

def strategy_holiday_drift(prices):
    """
    Long SPY day before market holiday.
    """
    if 'SPY' not in prices.columns: return None
    
    spy = prices['SPY'].pct_change()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=prices.index[0], end=prices.index[-1])
    
    # Shift holidays to find T-1 trading day
    # We want the return occurring ON the pre-holiday.
    # Which means we buy T-2 Close and Sell T-1 Close?
    # Or Buy T-1 Open Sell T-1 Close?
    # Anomaly is usually Pre-Holiday Drift (Close to Close of day before).
    
    # Find all T-1 dates
    # But holidays are non-trading days.
    # So we need to check if (Date + 1) is a holiday? Or Date + X is holiday?
    
    # Easiest way: merge holiday list.
    
    # Logic:
    # 1. Get all valid trading days.
    # 2. Check if Next Trading Day is > 1 calendar day away? (Weekends)
    # 3. Check if Next Calendar Day is Holiday?
    
    # Better: Use pandas offsets
    # CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # But complicated.
    
    # Simple check:
    # If tomorrow is a holiday.
    
    signal = pd.Series(0.0, index=prices.index)
    
    # Create a set of holidays
    hol_set = set(holidays.date)
    
    # Iterate (slow but safe)
    # Strategy: Long if T+1 is a holiday
    
    dates = prices.index
    for i in range(len(dates) - 1):
        curr_date = dates[i]
        next_date = dates[i+1] # The next trading day
        
        # Check gap
        # If gap > 1 day, was it a Monday (weekend gap) or a generic holiday?
        # Holiday Drift specifies: "Day before a holiday".
        # If Holiday is Friday, we buy Thursday. Thursday is T-1.
        
        # Let's check against hol_set
        # We look ahead 1 calendar day
        morrow = curr_date + pd.Timedelta(days=1)
        if morrow.date() in hol_set:
            signal.iloc[i] = 1.0
            
    return (signal * spy).fillna(0)

# =============================================================================
# 3. ANALYSIS
# =============================================================================

def analyze_seasonal(prices):
    print("\n📅 SEASONAL STRATEGY RESULTS:")
    print("-" * 75)
    
    spy_bh = prices['SPY'].pct_change().dropna()
    
    # Strategies
    r_sim = strategy_sell_in_may(prices)
    r_santa = strategy_santa_rally(prices)
    r_hol = strategy_holiday_drift(prices)
    
    stats = []
    
    def calc_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        stats.append({'Name': name, 'Sharpe': sharpe, 'Ann': ann, 'Conf': 'Low'})
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f}")

    calc_stats(spy_bh, "SPY (Buy & Hold)")
    print("-" * 75)
    if r_sim is not None: calc_stats(r_sim, "Sell in May (Winter)")
    if r_santa is not None: calc_stats(r_santa, "Santa Claus Rally")
    if r_hol is not None: calc_stats(r_hol, "Pre-Holiday Drift")
    
    print("-" * 75)
    
    # Verdict
    # Did any beat SPY Sharpe?
    best_sharpe = 0
    best_name = ""
    for s in stats:
        if s['Name'] != "SPY (Buy & Hold)" and s['Sharpe'] > best_sharpe:
            best_sharpe = s['Sharpe']
            best_name = s['Name']
            
    spy_sharpe = stats[0]['Sharpe']
    
    if best_sharpe > spy_sharpe + 0.1:
        print(f"✅ ANOMALY CONFIRMED: {best_name} works!")
    else:
        print("❌ SEASONALITY FADE: None of the calendar tricks outperformed Buy & Hold significantly.")

if __name__ == "__main__":
    prices = fetch_data()
    analyze_seasonal(prices)
