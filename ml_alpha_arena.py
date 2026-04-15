"""
ML Alpha Arena (Supervised Learning)
====================================

Trains multiple regression models to predict FUTURE returns based on PAST data.
Models:
1. Lasso (L1): Feature selection (sparsity).
2. Ridge (L2): Robustness to noise.
3. Random Forest: Non-linear patterns.
4. Gradient Boosting: Ensemble power.

Metric: Directional Accuracy (Did we predict Up/Down correctly?)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def load_and_prep_data():
    try:
        df = pd.read_csv('data/polymarket_real.csv')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        
        # Pivot to get time series per question
        price_matrix = df.pivot_table(index='datetime', columns='Question', values='Price')
        price_matrix = price_matrix.resample('1h').last().ffill()
        
        return price_matrix
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_features(series):
    """Generate ML Features for a single price series"""
    df = pd.DataFrame({'Price': series})
    
    # Returns
    df['Ret_1'] = df['Price'].pct_change(1)
    df['Ret_3'] = df['Price'].pct_change(3)
    
    # Lags
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Ret_1'].shift(i)
        
    # Volatility
    df['Vol_5'] = df['Ret_1'].rolling(5).std()
    
    # Momentum
    df['MA_5'] = df['Price'].rolling(5).mean()
    df['Dist_MA'] = (df['Price'] - df['MA_5']) / df['MA_5']
    
    # TARGET: Next Period Return
    df['Target'] = df['Ret_1'].shift(-1)
    
    df = df.dropna()
    return df

def main():
    print("🤖 ML ALPHA ARENA")
    print("=================")
    
    prices = load_and_prep_data()
    if prices is None: return
    
    print(f"   Loaded {len(prices.columns)} markets.")
    
    # We will train specific models for EACH market (simplest approach) 
    # OR one global model. Let's try Global Model (more data).
    
    all_X = []
    all_y = []
    
    print("   Feature Engineering...")
    for q in prices.columns:
        feat_df = create_features(prices[q])
        if len(feat_df) < 20: continue # Skip short history
        
        X = feat_df.drop(columns=['Price', 'Target', 'MA_5']) # Keep only stationary features
        y = feat_df['Target']
        
        all_X.append(X)
        all_y.append(y)
        
    if not all_X:
        print("❌ Not enough history for ML.")
        return
        
    X_full = pd.concat(all_X)
    y_full = pd.concat(all_y)
    
    # Clip outliers (Prediction markets can jump 100%)
    # X_full = X_full.clip(lower=-1.0, upper=1.0) 
    
    # Train/Test Split (Time based would be better, but random for now is okay for arena)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, shuffle=False)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "OLS Baseline": LinearRegression(),
        "Lasso (L1)": Lasso(alpha=0.01),
        "Ridge (L2)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        "Gradient Boost": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    }
    
    print(f"\n🥊 TRAINING {len(models)} MODELS on {len(X_train)} samples...")
    print(f"{'Model':<20} {'MSE (Error)':<12} {'Dir Acc %':<12} {'Alpha?'}")
    print("-" * 60)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, pred)
        
        # Directional Accuracy (Sign Match)
        # Avoid zero division
        correct_dir = np.sign(pred) == np.sign(y_test)
        acc = np.mean(correct_dir) * 100
        
        verdict = "⚪ Noise"
        if acc > 52.0: verdict = "🟢 Edge"
        if acc > 55.0: verdict = "🚀 ALPHA"
        
        print(f"{name:<20} {mse:<12.5f} {acc:<12.1f} {verdict}")
        
        # If Lasso, print features
        if "Lasso" in name:
            coefs = model.coef_
            print(f"   [Lasso Feats]: {dict(zip(X_full.columns, np.round(coefs, 4)))}")

    print("\n📝 NOTE: 'Dir Acc > 50%' acts as the Casino Edge.")

if __name__ == "__main__":
    main()
