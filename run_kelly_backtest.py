"""
Run backtest to calculate Kelly Criterion for the trading strategy.
"""
import sys
import pandas as pd
import numpy as np

# Import the backtester from ml_engine (uses the Lasso strategy)
from ml_engine import Backtester
from external_data import fetch_market_data_pool, TICKER_MAP

def main():
    print("=" * 60)
    print("   KELLY CRITERION BACKTEST")
    print("=" * 60)
    
    # Fetch market data
    print("\n📊 Fetching market data...")
    try:
        # Use a subset of major assets for faster backtest
        assets = [
            "S&P 500", "Nasdaq 100", "Russell 2000", "Dow Jones",
            "XLF (Financials)", "XLK (Technology)", "XLE (Energy)", 
            "XLV (Health Care)", "GLD (Gold ETF)", "TLT (20Y+ Treasury)"
        ]
        price_data = fetch_market_data_pool(assets, timeframe="2y")
        
        if price_data is None or price_data.empty:
            print("❌ No market data available. Using synthetic data for demo.")
            raise ValueError("Empty data")
    except Exception as e:
        print(f"⚠️ Error fetching data: {e}")
        print("Using synthetic data for demonstration.")
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=500, freq='B')
        assets_synthetic = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        returns = np.random.randn(500, len(assets_synthetic)) * 0.01 + 0.0003
        prices = 100 * np.exp(np.cumsum(returns, axis=0))
        price_data = pd.DataFrame(prices, index=dates, columns=assets_synthetic)
    
    print(f"✅ Loaded {len(price_data)} days of data for {len(price_data.columns)} assets")
    print(f"   Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    
    # Run backtest
    print("\n🚀 Running backtest with Lasso Momentum Strategy...")
    backtester = Backtester(initial_capital=10000.0, transaction_cost_pct=0.001)
    
    def progress(p):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"\r   [{bar}] {p*100:.0f}%", end="", flush=True)
    
    results = backtester.run(price_data, window_size=40, progress_callback=progress)
    print()  # Newline after progress bar
    
    if "error" in results:
        print(f"❌ Backtest failed: {results['error']}")
        return
    
    # Display results
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print("   BACKTEST RESULTS")
    print("=" * 60)
    
    print("\n📈 Performance Metrics:")
    print(f"   Total Return:      {metrics['Total Return']:>10.2%}")
    print(f"   CAGR:              {metrics['CAGR']:>10.2%}")
    print(f"   Sharpe Ratio:      {metrics['Sharpe']:>10.2f}")
    print(f"   Volatility:        {metrics['Volatility']:>10.2%}")
    
    # The ml_engine Backtester doesn't track individual trades the same way
    # Use the equity curve to estimate Kelly from Sharpe
    daily_returns = results['equity_curve'].pct_change().dropna()
    
    # Estimate win rate from daily returns
    winning_days = (daily_returns > 0).sum()
    total_days = len(daily_returns)
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    avg_win = daily_returns[daily_returns > 0].mean() if winning_days > 0 else 0
    avg_loss = abs(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).sum() > 0 else 0
    
    if avg_loss > 0:
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    else:
        win_loss_ratio = float('inf')
        kelly = 1.0
    
    print("\n🎯 Trade Statistics (Daily):")
    print(f"   Winning Days:      {winning_days:>10}")
    print(f"   Losing Days:       {total_days - winning_days:>10}")
    print(f"   Win Rate:          {win_rate:>10.1%}")
    print(f"   Avg Daily Win:     {avg_win:>10.3%}")
    print(f"   Avg Daily Loss:    {avg_loss:>10.3%}")
    print(f"   Win/Loss Ratio:    {win_loss_ratio:>10.2f}")
    
    print("\n" + "=" * 60)
    print("   KELLY CRITERION")
    print("=" * 60)
    print(f"\n   Full Kelly:        {kelly:>10.1%}")
    print(f"   Half Kelly:        {kelly/2:>10.1%}")
    print(f"   Quarter Kelly:     {kelly/4:>10.1%}")
    
    # Interpretation
    print("\n📊 Interpretation:")
    if kelly > 1:
        print(f"   ⚡ Kelly suggests {kelly:.1%} allocation (>100% = use leverage)")
        print(f"   ✅ Recommended: Use Half-Kelly ({kelly/2:.1%}) for safety")
    elif kelly > 0:
        print(f"   ✅ Kelly suggests {kelly:.1%} of capital per position")
        print(f"   ✅ Recommended: Use Half-Kelly ({kelly/2:.1%}) for safety")
    else:
        print(f"   ⚠️ Negative Kelly ({kelly:.1%}) - strategy has negative edge")
        print(f"   ❌ Do not trade with real capital until strategy is improved")
    
    print("\n" + "=" * 60)
    
    return results

if __name__ == "__main__":
    main()
