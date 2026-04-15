"""
ERP Inventory Alpha Hunt
========================

Reconstructing historical inventory levels from raw transaction logs (IV30300).
Hypothesis: Internal Inventory Buildup predicts Future Stock Returns (Supply/Demand Signal).

Tables:
- IV30300 (Transaction History): DOCDATE, TRXQTY, UNITCOST

Process:
1. Extract History.
2. Reconstruct rolling Inventory Balances.
3. Correlate trend (YoY Change) with External Ag Stocks (NTR, MOS).

RUN: python erp_inventory_hunt.py
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
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={secrets['server']};"
            f"DATABASE={secrets['database']};"
            f"UID={secrets['username']};"
            f"PWD={secrets['password']}"
        )
        return pyodbc.connect(conn_str)
    except Exception as e:
        print(f"⚠️ DB Connection Failed: {e}")
        return None

def fetch_inventory_history():
    print("🏭 Mining IV30300 (Transaction History)...")
    conn = get_db_connection()
    
    if conn:
        query = """
        SELECT 
            DOCDATE,
            ITEMNMBR,
            TRXQTY,
            UNITCOST
        FROM 
            IV30300
        WHERE 
            DOCDATE >= '2018-01-01'
        ORDER BY 
            DOCDATE ASC
        """
        try:
            df = pd.read_sql(query, conn)
            conn.close()
            print(f"   Extracted {len(df)} transactions.")
            return df
        except Exception as e:
            print(f"   Query Failed: {e}")
            if conn: conn.close()
    
    # Fallback Dummy Data (for Simulation)
    print("   ⚠️ Using Dummy Data for Simulation")
    dates = pd.date_range(start='2018-01-01', end=datetime.now(), freq='D')
    # Simulate random transactions
    n = len(dates) * 2
    dummy_dates = np.random.choice(dates, n)
    # Seasonal effect
    months = pd.to_datetime(dummy_dates).month
    qty = np.random.randn(n) * 100
    # Boost qty in Spring (Seasonal Buildup)
    qty += np.where((months >= 3) & (months <= 5), 200, 0)
    
    df = pd.DataFrame({
        'DOCDATE': dummy_dates,
        'TRXQTY': qty,
        'UNITCOST': 10 + np.random.randn(n)
    })
    return df

# =============================================================================
# 2. SIGNAL GENERATION
# =============================================================================

def reconstruct_timeseries(txn_df):
    print("⏳ Reconstructing Historical Inventory Levels...")
    
    # Convert dates
    txn_df['DOCDATE'] = pd.to_datetime(txn_df['DOCDATE'])
    
    # Calculate Value Change per txn
    txn_df['ValChange'] = txn_df['TRXQTY'] * txn_df['UNITCOST']
    
    # Group by Month
    monthly_flow = txn_df.set_index('DOCDATE').resample('M')['ValChange'].sum()
    
    # Cumulative Sum to get Balance (assuming start 0 or finding relative trend)
    # We care about *Change* in inventory, mostly.
    # But lets define Balance.
    inventory_balance = monthly_flow.cumsum()
    
    # Signal: Year-over-Year Change in Inventory Balance
    # (Are we holding more stuff than last year?)
    inv_yoy = inventory_balance.pct_change(12).fillna(0)
    
    # Signal 2: Month-over-Month accumulation
    inv_mom = inventory_balance.pct_change(1).fillna(0)
    
    signal_df = pd.DataFrame({
        'Balance': inventory_balance,
        'Signal_YoY': inv_yoy,
        'Signal_MoM': inv_mom
    })
    
    return signal_df

# =============================================================================
# 3. EXTERNAL CORRELATION
# =============================================================================

def fetch_ag_stocks():
    print("📈 Fetching External Ag Stocks (NTR, MOS)...")
    tickers = ['NTR', 'MOS', 'CF', 'DBA'] 
    # DBA = Ag Futures ETF
    
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
           prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
           prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    return prices.resample('M').last()

def analyze_correlation(signal_df, stock_df):
    print("🔬 Correlating Internal Inventory vs Public Supply...")
    
    # Align dates
    # Signal is monthly end. Stocks monthly end.
    combined = stock_df.join(signal_df, how='inner')
    
    # Shift Signal?
    # Does Inventory Today predict Stock Returns Tomorrow?
    # Shift Stocks BACK (Return T+1) to align with Signal T?
    # Stocks next month return
    future_returns = stock_df.pct_change().shift(-1)
    
    combined_predictive = future_returns.join(signal_df, how='inner').dropna()
    
    print("\n📊 CORRELATION MATRIX (Signal vs Next Month Returns):")
    corr = combined_predictive.corr()
    print(corr.loc[['Signal_YoY', 'Signal_MoM'], ['NTR', 'MOS', 'CF', 'DBA']])
    
    # Backtest Simple Strategy
    # If Inventory Buildup (YoY > 0) -> Long Ag Stocks (Bullish Demand)
    # Else -> Cash
    
    ag_basket = stock_df.mean(axis=1).pct_change() # Basket Return
    
    # Strategy 1: Buildup = Bullish
    signal = (signal_df['Signal_YoY'] > 0).astype(int)
    # Shift signal to trade next month
    strat_ret = (signal.shift(1).reindex(ag_basket.index).fillna(0) * ag_basket)
    
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(12)
    
    print(f"\n🚜 STRATEGY RESULTS:")
    print(f"   Hypothesis: Inventory Buildup = Bullish Demand")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    
    if abs(sharpe) > 0.5:
        dir_str = "Positive" if sharpe > 0 else "Negative (Inventory Glut = Bearish)"
        print(f"   ✅ EDGE DETECTED ({dir_str})")
        return True
    else:
        print("   ❌ NO EDGE: Internal Inventory is noise.")
        return False

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # 1. Internal
    txn_df = fetch_inventory_history()
    signals = reconstruct_timeseries(txn_df)
    
    # 2. External
    stocks = fetch_ag_stocks()
    
    # 3. Test
    found_alpha = analyze_correlation(signals, stocks)
