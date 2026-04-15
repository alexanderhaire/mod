"""
Model Shootout: Finding the Best Brain
======================================

Comparing different ML algorithms on the Macro-Enhanced Dataset.
Universe: XLB, XLI, XLE, JNK, GLD, USMV, MTUM, FXA

Contenders:
1.  Random Forest (Current Champion) - Good at non-linearities, robust to noise.
2.  Gradient Boosting (GBM/XGBoost style) - Often higher accuracy, prones to overfitting.
3.  Lasso/Ridge (Linear Models) - Good if relationships are simple/linear (Macro often is).
4.  ElasticNet - Mix of L1/L2.

Metric: IC (Information Coefficient) and Directional Accuracy.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

TARGET_ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'FXA', 'USMV', 'MTUM']
MACRO_ASSETS = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    tickers = list(set(TARGET_ASSETS + MACRO_ASSETS))
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

def create_features(prices: pd.DataFrame) -> tuple:
    df = pd.DataFrame(index=prices.index)
    feature_cols = []
    
    # Macro
    if '^TNX' in prices.columns:
        df['rate_change'] = prices['^TNX'].diff(20)
        df['rate_trend'] = prices['^TNX'] - prices['^TNX'].rolling(60).mean()
        feature_cols += ['rate_change', 'rate_trend']
        
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        df['credit_spread'] = prices['JNK'] / prices['IEF']
        feature_cols += ['credit_spread']
        
    if 'UUP' in prices.columns:
        df['dollar_vol'] = prices['UUP'].pct_change().rolling(20).std()
        feature_cols += ['dollar_vol']
        
    # Asset specific
    asset_map = {}
    for t in TARGET_ASSETS:
        if t not in prices.columns: continue
        r = prices[t].pct_change()
        m1 = f'{t}_mom_1m'
        m3 = f'{t}_mom_3m'
        v1 = f'{t}_vol_1m'
        df[m1] = r.rolling(20).mean()
        df[m3] = r.rolling(60).mean()
        df[v1] = r.rolling(20).std()
        asset_map[t] = [m1, m3, v1] + feature_cols
        
    # Target
    target = prices.pct_change().shift(-5).rolling(5).sum() # 5 day target
    
    return df, target, asset_map

def run_shootout():
    print("=" * 80)
    print("   MODEL SHOOTOUT: RF vs GBM vs LASSO")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    features, targets, asset_map = create_features(prices)
    
    # Contenders
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        'Grad Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        'Lasso (Linear)': make_pipeline(StandardScaler(), Lasso(alpha=0.001)),
        'Ridge (Linear)': make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    }
    
    results = {m: [] for m in models.keys()}
    
    print(f"   Testing on {len(TARGET_ASSETS)} assets...")
    
    for ticker in TARGET_ASSETS:
        if ticker not in prices.columns: continue
        
        # Data Prep
        cols = asset_map[ticker]
        X = features[cols]
        y = targets[ticker]
        
        # Merge and dropna
        data = pd.concat([X, y], axis=1).dropna()
        if len(data) < 250: continue
        
        X = data[cols]
        y = data.iloc[:, -1]
        
        # Split
        split = int(len(data) * 0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Test each model
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            ic = spearmanr(preds, y_test)[0]
            results[name].append(ic)
            
    print("\n   === AVERAGE PREDICTIVE POWER (IC) ===")
    sorted_res = sorted([(name, np.mean(ics)) for name, ics in results.items()], key=lambda x: x[1], reverse=True)
    
    for name, avg_ic in sorted_res:
        print(f"   {name:<15} : {avg_ic:.4f}")
        
    winner = sorted_res[0]
    print(f"\n   🏆 WINNER: {winner[0]} (IC {winner[1]:.4f})")
    
    if winner[0] != 'Random Forest':
        print("   ✅ UPGRADE RECOMMENDED: Switch ML Engine.")
    else:
        print("   ✅ KEEP CURRENT: Random Forest is still king.")

if __name__ == "__main__":
    run_shootout()
