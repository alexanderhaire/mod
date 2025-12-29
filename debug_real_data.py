"""Debug script to check why model isn't trading on real data"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine import PredictiveAlphaEngine
from external_data import fetch_market_data_pool

# Minimal test with just a few ETFs
TEST_ASSETS = ["SPY (S&P 500)", "QQQ (Nasdaq 100)", "XLE (Energy)"]

print("Fetching data...")
pool = fetch_market_data_pool(TEST_ASSETS)

# Build price DF
price_data = {}
for asset, info in pool.items():
    if info.get("data") and len(info["data"]) > 0:
        dates = [d["date"] for d in info["data"]]
        prices = [d["price_index"] for d in info["data"]]
        price_data[asset] = pd.Series(prices, index=pd.to_datetime(dates), name=asset)

df = pd.DataFrame(price_data).ffill().dropna()
print(f"Loaded {len(df)} days, {len(df.columns)} assets")

# Create engine with 12 features
assets = list(df.columns)
engine = PredictiveAlphaEngine(assets)

# Train on first 200 days
print("\n=== Training Phase ===")
for t in range(40, 200):
    for asset in assets:
        hist_slice = df[asset].values[max(0, t-32) : t]
        if len(hist_slice) >= 31:
            feats = engine._extract_features(hist_slice)
            target_ret = (df[asset].iloc[t] - df[asset].iloc[t-1]) / df[asset].iloc[t-1]
            engine.models[asset].update(feats, target_ret)

print(f"Training complete. Models have {engine.models[assets[0]].n_updates} updates")

# Now check predictions/confidence on day 201+
print("\n=== Prediction Phase ===")
for t in [200, 250, 300, 350]:
    if t >= len(df):
        continue
        
    print(f"\n--- Day {t} ({df.index[t].date()}) ---")
    total_conf = 0
    for asset in assets:
        curr_slice = df[asset].values[max(0, t-31) : t+1]
        feats = engine._extract_features(curr_slice)
        pred = engine.models[asset].predict(feats)
        conf = engine.models[asset].get_confidence_score()
        total_conf += conf
        print(f"  {asset}: pred={pred:+.4f}, conf={conf:.4f}, feats_sum={np.sum(np.abs(feats)):.4f}")
    
    avg_conf = total_conf / len(assets)
    print(f"  Avg confidence: {avg_conf:.4f}")
    if avg_conf < 0.10:
        print(f"  ⚠️ Would be BLOCKED by confidence gate (threshold=0.10)")
    else:
        print(f"  ✅ Would TRADE")
