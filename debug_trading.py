"""
Debug script to understand why model isn't trading.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine import Backtester, PredictiveAlphaEngine
import logging

logging.basicConfig(level=logging.WARNING)

def generate_dummy_data(days=100, assets=5, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=days, freq="B")
    data = {}
    for i in range(assets):
        prices = [100.0]
        drift = np.random.uniform(-0.0001, 0.0003)
        vol = np.random.uniform(0.008, 0.02)
        for _ in range(days - 1):
            change = np.random.normal(drift, vol)
            prices.append(prices[-1] * (1 + change))
        data[f"Asset_{i}"] = prices
    return pd.DataFrame(data, index=dates)

def debug_run():
    print("Generating data...")
    df = generate_dummy_data(days=100, assets=5, seed=42)
    
    # Manually step through what the backtester does
    assets = df.columns.tolist()
    ml_engine = PredictiveAlphaEngine(assets)
    
    returns_df = df.pct_change().fillna(0)
    
    window_size = 30
    
    # Simulate a few steps
    for t in range(window_size, min(window_size + 10, len(df) - 1)):
        print(f"\n--- Day {t} ---")
        
        # Train models (if t > window_size)
        if t > window_size:
            target_returns = returns_df.iloc[t]
            for asset in assets:
                hist_slice = df[asset].values[max(0, t-12) : t]
                if len(hist_slice) > 10:
                    feats = ml_engine._extract_features(hist_slice)
                    actual_ret = target_returns[asset]
                    ml_engine.models[asset].update(feats, actual_ret)
        
        # Predict
        predicted_returns = {}
        confidence_scaled_returns = {}
        avg_confidence = 0.0
        
        for asset in assets:
            curr_slice = df[asset].values[max(0, t-11) : t+1]
            feats = ml_engine._extract_features(curr_slice)
            pred_ret = ml_engine.models[asset].predict(feats)
            predicted_returns[asset] = pred_ret
            
            confidence = ml_engine.models[asset].get_confidence_score()
            avg_confidence += confidence
            scaled_pred = pred_ret * (0.1 + 0.9 * confidence)
            confidence_scaled_returns[asset] = scaled_pred
            
            print(f"  {asset}: pred={pred_ret:.6f}, conf={confidence:.3f}, scaled={scaled_pred:.6f}")
        
        avg_confidence /= len(assets)
        print(f"  AVG CONFIDENCE: {avg_confidence:.3f}")
        print(f"  THRESHOLD: 0.10")
        print(f"  WILL TRADE: {avg_confidence >= 0.10}")
        
        # Check signal allocation
        if avg_confidence >= 0.10:
            raw_signals = {}
            for asset in assets:
                pred = confidence_scaled_returns[asset]
                conf = ml_engine.models[asset].get_confidence_score()
                raw_signals[asset] = np.sign(pred) * conf * abs(pred) * 100
            
            total_signal = sum(abs(s) for s in raw_signals.values())
            print(f"  TOTAL SIGNAL: {total_signal:.6f}")
            
            if total_signal > 1e-6:
                weights = {a: s / total_signal for a, s in raw_signals.items()}
                print(f"  WEIGHTS: {weights}")
            else:
                print(f"  WEIGHTS: ALL ZERO (no signal)")

if __name__ == "__main__":
    debug_run()
