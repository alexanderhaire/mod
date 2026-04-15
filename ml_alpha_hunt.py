"""
ML Alpha Hunt (The Nuclear Option) - REFINED
=============================================

Aggregating ALL previously tested data sources:
- Equities (SPY, QQQ, IWM)
- Macro (TLT, GLD, USO, UUP)
- Crypto (BTC, ETH)
- Volatility (VIX proxy)

Training Non-Linear ML Models (Random Forest & GBM) to predict SPY Next Day Return.
FIX: Drops weekends/holidays to prevent "predicting 0 return on saturday" data leakage.

RUN: python ml_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA AGGREGATION
# =============================================================================

def fetch_master_dataset():
    print("🧠 Fetching MASTER DATASET (Equities + Macro + Crypto)...")
    
    tickers = {
        'SPY': 'SPY', 
        'QQQ': 'QQQ', 
        'IWM': 'IWM',       # Small Caps
        'TLT': 'TLT',       # Rates
        'GLD': 'GLD',       # Gold
        'USO': 'USO',       # Oil
        'UUP': 'UUP',       # Dollar
        'BTC': 'BTC-USD',   # Crypto Risk
        'ETH': 'ETH-USD'    # Crypto Beta
    }
    
    # Download
    data = yf.download(list(tickers.values()), start='2015-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try getting 'Adj Close'
            if 'Adj Close' in data.columns.get_level_values(0):
                adj_close = data['Adj Close']
            else:
                 # Fallback to 'Close'
                adj_close = data['Close']
        except:
             adj_close = data['Close']
    else:
        adj_close = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]
        
    # Rename columns to friendly names
    reverse_map = {v: k for k, v in tickers.items()}
    adj_close = adj_close.rename(columns=reverse_map)
    
    # FIX: Align to SPY trading days to avoid Weekend artifacts
    if 'SPY' in adj_close.columns:
        spy_valid_days = adj_close['SPY'].dropna().index
        # Forward fill first to fill crypto gaps on weekends if any (so Monday has data)
        adj_close = adj_close.ffill()
        # Reindex to only valid SPY days
        df = adj_close.loc[spy_valid_days]
    else:
        df = adj_close.dropna()
    
    print(f"   Data: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Assets: {list(df.columns)}")
    print(f"   Rows: {len(df)}")
    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def create_features(df):
    print("🛠️ Generating Technical & Macro Features...")
    features = pd.DataFrame(index=df.index)
    
    # 1. Momentum & Trends (for ALL assets)
    for col in df.columns:
        # Returns
        features[f'{col}_Ret_1d'] = df[col].pct_change(1)
        features[f'{col}_Ret_5d'] = df[col].pct_change(5)
        features[f'{col}_Ret_21d'] = df[col].pct_change(21)
        
        # Volatility
        features[f'{col}_Vol_21d'] = df[col].pct_change().rolling(21).std()
        
        # Distance to SMA (Trend)
        sma50 = df[col].rolling(50).mean()
        features[f'{col}_DistSMA50'] = (df[col] / sma50) - 1
        
    # 2. Structural Features
    features['DayOfWeek'] = df.index.dayofweek
    features['Month'] = df.index.month
    features['IsMonthEnd'] = df.index.is_month_end.astype(int)
    
    # 3. Cross-Asset Interact terms
    if 'USO_Ret_21d' in features.columns and 'SPY_Ret_21d' in features.columns:
        features['Oil_vs_SPY'] = features['USO_Ret_21d'] - features['SPY_Ret_21d']
    if 'TLT_Ret_21d' in features.columns and 'QQQ_Ret_21d' in features.columns:
        features['Rates_vs_Tech'] = features['TLT_Ret_21d'] - features['QQQ_Ret_21d']
    
    # TARGET: Next Day SPY Direction (1 = Up, 0 = Down)
    # Price(T+1) > Price(T)
    target = (df['SPY'].shift(-1) > df['SPY']).astype(int)
    
    # Drop NaNs
    dataset = features.join(target.rename('Target')).dropna()
    return dataset

# =============================================================================
# 3. ML MODELING
# =============================================================================

def train_and_validate(dataset):
    print("🤖 Training Random Forest & Gradient Boosting (Walk-Forward)...")
    
    X = dataset.drop('Target', axis=1)
    y = dataset['Target']
    
    # Test Period: 2023-01-01 to Present
    split_date = '2023-01-01'
    X_train = X[X.index < split_date]
    y_train = y[y.index < split_date]
    X_test = X[X.index >= split_date]
    y_test = y[y.index >= split_date]
    
    print(f"   Train Size: {len(X_train)} days")
    print(f"   Test Size:  {len(X_test)} days (Since {split_date})")
    
    # Check Class Balance
    print("\n⚖️ TEST SET CLASS BALANCE:")
    print(y_test.value_counts(normalize=True))
    baseline = y_test.value_counts(normalize=True).max()
    print(f"   Baseline Accuracy (Always Predict Majority): {baseline:.2%}")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n👉 Model: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        
        print(f"   Accuracy:  {acc:.2%}")
        if acc > baseline + 0.02: # 2% edge
            print("   ✅ BEATS MARKET (Slightly)")
        else:
            print("   ❌ NO EDGE vs Baseline")
            
        print("   Classification Report:")
        print(classification_report(y_test, preds, zero_division=0))
        
        if name == 'Random Forest':
            print("   Top Features:")
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
            print(imp)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("☢️ ML ALPHA HUNT: REFINED KITCHEN SINK")
    print("="*60)
    
    # 1. Get Data
    df = fetch_master_dataset()
    
    # 2. Build Dataset
    features = create_features(df)
    
    # 3. Train & Test
    train_and_validate(features)
    
    print("\n" + "=" * 60)
