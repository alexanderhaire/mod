"""Backtesting Framework for ML Trading Strategies

Walk-forward backtesting with realistic transaction costs, slippage, and risk management.
"""

import logging
from typing import Any, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


class Backtester:
    """Walk-forward backtesting engine with transaction cost modeling."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,  # 5 basis points
        slippage_bps: float = 3.0,  # 3 basis points market impact
        max_position_pct: float = 0.10,  # 10% max per position
        stop_loss_pct: float = 0.15  # 15% stop loss
    ):
        """
        Args:
            initial_capital: Starting cash
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage/market impact in basis points
            max_position_pct: Maximum allocation per asset
            stop_loss_pct: Stop loss threshold
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        
        # State tracking
        self.cash = initial_capital
        self.positions: dict[str, float] = {}  # asset -> quantity
        self.entry_prices: dict[str, float] = {}  # asset -> entry price
        self.trade_log: list[dict] = []
        self.equity_curve: list[float] = []
        self.dates: list[datetime] = []
        
    def calculate_transaction_cost(self, price: float, quantity: float) -> float:
        """Calculate total transaction cost including slippage."""
        notional = abs(price * quantity)
        total_cost_bps = self.transaction_cost_bps + self.slippage_bps
        return notional * (total_cost_bps / 10000)
    
    def execute_trade(self, date: datetime, asset: str, target_weight: float, current_price: float):
        """
        Execute a trade to rebalance to target weight.
        
        Args:
            date: Trade date
            asset: Asset ticker
            target_weight: Desired portfolio weight (0-1)
            current_price: Current market price
        """
        # Calculate current portfolio value
        portfolio_value = self.cash + sum(
            self.positions.get(a, 0) * current_price 
            for a in self.positions
        )
        
        # Target dollar amount
        target_dollars = portfolio_value * min(target_weight, self.max_position_pct)
        current_dollars = self.positions.get(asset, 0) * current_price
        
        # Required trade
        trade_dollars = target_dollars - current_dollars
        trade_quantity = trade_dollars / current_price
        
        if abs(trade_quantity) < 0.01:  # Skip tiny trades
            return
        
        # Calculate costs
        cost = self.calculate_transaction_cost(current_price, trade_quantity)
        
        # Execute
        self.cash -= (trade_quantity * current_price + cost)
        self.positions[asset] = self.positions.get(asset, 0) + trade_quantity
        
        if trade_quantity > 0:
            self.entry_prices[asset] = current_price
        
        # Log trade
        self.trade_log.append({
            'date': date,
            'asset': asset,
            'action': 'BUY' if trade_quantity > 0 else 'SELL',
            'quantity': abs(trade_quantity),
            'price': current_price,
            'cost': cost
        })
        
    def check_stop_loss(self, date: datetime, asset: str, current_price: float):
        """Check if stop loss triggered."""
        if asset not in self.entry_prices:
            return False
            
        entry_price = self.entry_prices[asset]
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct < -self.stop_loss_pct:
            LOGGER.warning(f"STOP LOSS: {asset} down {pnl_pct:.1%}")
            # Close position
            self.execute_trade(date, asset, 0.0, current_price)
            return True
        
        return False
    
    def update_equity(self, date: datetime, prices: dict[str, float]):
        """Update equity curve."""
        portfolio_value = self.cash
        
        for asset, quantity in self.positions.items():
            if asset in prices:
                portfolio_value += quantity * prices[asset]
        
        self.equity_curve.append(portfolio_value)
        self.dates.append(date)
        
    def run_backtest(
        self,
        price_history: pd.DataFrame,
        signal_generator: callable,
        rebalance_frequency: int = 5  # days
    ) -> pd.DataFrame:
        """
        Run walk-forward backtest.
        
        Args:
            price_history: DataFrame with Date index, asset columns
            signal_generator: Function that takes (date, prices) -> dict of weights
            rebalance_frequency: Days between rebalances
            
        Returns:
            DataFrame with backtest results
        """
        LOGGER.info(f"Starting backtest from {price_history.index[0]} to {price_history.index[-1]}")
        
        for i, date in enumerate(price_history.index):
            current_prices = price_history.loc[date].to_dict()
            
            # Check stop losses
            for asset in list(self.positions.keys()):
                if asset in current_prices:
                    self.check_stop_loss(date, asset, current_prices[asset])
            
            # Rebalance periodically
            if i % rebalance_frequency == 0:
                # Get signals from ML model
                target_weights = signal_generator(date, price_history.loc[:date])
                
                # Execute trades
                for asset, weight in target_weights.items():
                    if asset in current_prices:
                        self.execute_trade(date, asset, weight, current_prices[asset])
            
            # Update equity curve
            self.update_equity(date, current_prices)
        
        # Calculate performance metrics
        results = self._calculate_metrics()
        return results
    
    def _calculate_metrics(self) -> dict[str, Any]:
        """Calculate performance statistics including Kelly Criterion."""
        equity = pd.Series(self.equity_curve, index=self.dates)
        returns = equity.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return / (len(equity) / 252)  # Assuming daily data
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # --- FIXED: Proper Win Rate & Kelly Criterion Calculation ---
        # Track P&L for each completed round-trip trade
        wins = []
        losses = []
        
        # Match BUY trades with subsequent SELL trades by asset
        open_positions = {}  # asset -> (entry_price, quantity)
        
        for trade in self.trade_log:
            asset = trade['asset']
            action = trade['action']
            price = trade['price']
            quantity = trade['quantity']
            
            if action == 'BUY':
                # Open or add to position
                if asset in open_positions:
                    # Average into position
                    old_price, old_qty = open_positions[asset]
                    new_qty = old_qty + quantity
                    avg_price = (old_price * old_qty + price * quantity) / new_qty
                    open_positions[asset] = (avg_price, new_qty)
                else:
                    open_positions[asset] = (price, quantity)
                    
            elif action == 'SELL' and asset in open_positions:
                # Close position and calculate P&L
                entry_price, _ = open_positions[asset]
                pnl_pct = (price - entry_price) / entry_price
                
                if pnl_pct > 0:
                    wins.append(pnl_pct)
                else:
                    losses.append(abs(pnl_pct))
                
                # Remove position (or reduce if partial - simplified to full close)
                del open_positions[asset]
        
        # Calculate statistics
        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Kelly Criterion: K = (p * b - q) / b
        # Where p = win rate, q = loss rate, b = avg_win / avg_loss
        if avg_loss > 0 and total_trades > 0:
            win_loss_ratio = avg_win / avg_loss
            kelly_criterion = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        else:
            win_loss_ratio = 0
            kelly_criterion = 0
        
        # Half-Kelly (safer position sizing)
        half_kelly = kelly_criterion / 2
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'kelly_criterion': kelly_criterion,
            'half_kelly': half_kelly,
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'final_equity': equity.iloc[-1],
            'equity_curve': equity,
            'trade_log': pd.DataFrame(self.trade_log)
        }
