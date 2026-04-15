"""
Sales Alpha Hunt (The Peter Lynch Strategy)
===========================================

Testing if Internal Sales Velocity (SOP30200) predicts External Ag Stocks.
Hypothesis: "Local Demand" = "Global Demand". High internal sales -> Buy NTR/MOS.

Tables:
- SOP30200 (Sales History): DOCDATE, XTNDPRCE (Revenue)

Process:
1. Extract monthly sales revenue.
2. Calculate YoY Growth.
3. Trade External Stocks based on Internal Growth.

RUN: python sales_alpha_hunt.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pyodbc
from secrets_loader import load_sql_secrets as load_secrets
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATABASE EXTRACTION
# =============================================================================

def get_db_connection():
    try:
        secrets = load_secrets()
        # Handle potential key variations based on previous error
        server = secrets.get('server')
        database = secrets.get('database')
        user = secrets.get('username') or secrets.get('user') or secrets.get('uid')
        pwd = secrets.get('password') or secrets.get('pwd')
        
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={pwd}"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"⚠️ DB Connection Failed: {e}")
        return None

def fetch_sales_history():
    print("💰 Mining SOP30200 (Sales History)...")
    conn = get_db_connection()
    
    if conn:
        query = """
        SELECT 
            DOCDATE,
            XTNDPRCE
        FROM 
            SOP30200
        WHERE 
            DOCDATE >= '2018-01-01'
            AND SOPTYPE = 3 -- Invoices
        ORDER BY 
            DOCDATE ASC
        """
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"   Extracted {len(df)} invoices.")
            return df
        except Exception as e:
            print(f"   Query Failed: {e}")
            if conn: conn.close()
    
    # Fallback Dummy Data (Simulation)
    print("   ⚠️ Using Dummy Data for Simulation")
    dates = pd.date_range(start='2018-01-01', end=datetime.now(), freq='D')
    n = len(dates)
    
    # Simulate Sales Pattern
    # Seasonality (High in Q2 - Spring)
    months = dates.month
    base_sales = np.random.lognormal(mean=8, sigma=1, size=n) # Log-normal daily sales
    seasonality = np.where((months >= 3) & (months <= 5), 1.5, 1.0)
    sales = base_sales * seasonality
    
    df = pd.DataFrame({
        'DOCDATE': dates,
        'XTNDPRCE': sales
    })
    return df

# =============================================================================
# 2. SIGNAL GENERATION
# =============================================================================

def process_sales_signal(df):
    print("⏳ Processing Internal Demand Signal...")
    
    df['DOCDATE'] = pd.to_datetime(df['DOCDATE'])
    
    # Monthly Revenue
    monthly_sales = df.set_index('DOCDATE').resample('M')['XTNDPRCE'].sum()
    
    # YoY Growth Signal (Smoothing seasonality)
    # Are we selling more this May than last May?
    sales_yoy = monthly_sales.pct_change(12).fillna(0)
    
    # Trend Signal (3 month SMA of YoY)
    signal_trend = sales_yoy.rolling(3).mean()
    
    signal_df = pd.DataFrame({
        'Revenue': monthly_sales,
        'Growth_YoY': sales_yoy,
        'Signal': signal_trend
    })
    
    return signal_df

# =============================================================================
# 3. EXTERNAL CORRELATION
# =============================================================================

def fetch_ag_stocks():
    print("📈 Fetching External Ag Stocks (NTR, MOS)...")
    tickers = ['NTR', 'MOS', 'CF'] # Fertilizer Giants
    
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
     # Robust extraction
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    return prices.resample('M').last()

def backtest_sales_alpha(signal_df, stock_df):
    print("\n🔬 Testing 'Peter Lynch' Strategy...")
    
    # Align
    combined = stock_df.join(signal_df, how='inner')
    
    # Basket of Ag Stocks
    basket_ret = stock_df.pct_change().mean(axis=1)
    
    # Strategy:
    # If Internal Growth Trend > 0 (Growing Sales) -> Buy Ag Stocks
    # Else -> Cash
    
    signal = (combined['Signal'] > 0).astype(int)
    
    # Shift signal to trade NEXT month (using Last Month's sales data)
    strat_ret = signal.shift(1) * basket_ret
    strat_ret = strat_ret.dropna()
    basket_ret = basket_ret.dropna()
    
    # Metrics
    def get_stats(r):
        ann = r.mean() * 12
        vol = r.std() * np.sqrt(12)
        sharpe = ann / vol if vol > 0 else 0
        return ann, sharpe
        
    s_ann, s_sharpe = get_stats(strat_ret)
    b_ann, b_sharpe = get_stats(basket_ret)
    
    print(f"   Internal Sales Growth Strategy:")
    print(f"   Buy & Hold Ag: {b_ann:.1%} | Sharpe {b_sharpe:.2f}")
    print(f"   Internal Signal: {s_ann:.1%} | Sharpe {s_sharpe:.2f}")
    
    corr = combined['Signal'].corr(basket_ret.shift(-1))
    print(f"   Predictive Correlation: {corr:.3f}")
    
    if s_sharpe > b_sharpe + 0.1:
        print("   ✅ EDGE FOUND: Your sales predict the market!")
    else:
        print("   ❌ NO EDGE: Local sales don't move global stocks.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("💰 SALES ALPHA HUNT (INTERNAL DEMAND)")
    print("="*60)
    
    # 1. Internal
    sales_df = fetch_sales_history()
    signals = process_sales_signal(sales_df)
    
    # 2. External
    stocks = fetch_ag_stocks()
    
    # 3. Test
    backtest_sales_alpha(signals, stocks)
    
    print("\n" + "=" * 60)
