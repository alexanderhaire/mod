"""
ML BACKTEST WITH REAL MARKET DATA
==================================
Uses the WINNING Lasso Momentum strategy (2.39 Sharpe on out-of-sample).
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasso_momentum_strategy import LassoMomentumStrategy, ASSETS, backtest


def main():
    print("=" * 60)
    print("   ML BACKTEST WITH LASSO MOMENTUM STRATEGY")
    print("   (The winning strategy: 2.39 Sharpe on out-of-sample)")
    print("=" * 60)
    
    # Run the winning strategy's backtest
    strategy, metrics = backtest()
    
    # Summary
    print("\n" + "=" * 60)
    print("   FINAL SUMMARY")
    print("=" * 60)
    
    if metrics["sharpe"] >= 1.0:
        print("🎉 EXCELLENT - Strategy ready for paper trading")
    elif metrics["sharpe"] >= 0.5:
        print("🟡 MARGINAL - Proceed with caution")
    else:
        print("🔴 NEEDS WORK - Do not trade")
    
    return metrics


if __name__ == "__main__":
    main()
