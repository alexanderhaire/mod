
"""
Machine Learning & Optimization Engine
--------------------------------------
Provides continuous learning capabilities (Recursive Least Squares) and
portfolio optimization (Monte Carlo Efficient Frontier) using pure Numpy.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple

LOGGER = logging.getLogger(__name__)

class OnlineLinearRegressor:
    """
    Recursive Least Squares (RLS) filter for adaptive linear regression.
    Updates coefficients incrementally as new data arrives, allowing the model
    to "learn" and adapt to changing market correlations over time.
    
    Model: y = X * theta + noise
    """
    def __init__(self, n_features: int = 1, lambda_factor: float = 0.99, initial_variance: float = 1000.0):
        """
        Args:
            n_features: Number of independent variables (X).
            lambda_factor: Forgetting factor (0 < lambda <= 1). 
                           Lower values adapt faster but are nosier.
            initial_variance: Initial uncertainty in parameters (P matrix diagonal).
        """
        self.n_features = n_features
        self.lam = lambda_factor
        
        # Initialize Efficiency Matrix (Inverse Correlation Matrix)
        # P = (X^T * X)^-1
        self.P = np.eye(n_features) * initial_variance
        
        # Initialize Coefficients (theta)
        self.theta = np.zeros((n_features, 1))
        
        # Stats for monitoring "Learning"
        self.error_history = []
        self.n_updates = 0
        
    def predict(self, x: np.ndarray) -> float:
        """Predict y given x."""
        x = x.reshape(self.n_features, 1)
        return float(np.dot(self.theta.T, x))
        
    def update(self, x: np.ndarray, y: float) -> float:
        """
        Update model parameters with a new observation (x, y).
        Returns the prediction error (prior to update).
        """
        x = x.reshape(self.n_features, 1)
        
        # 1. Calculate prediction using old parameters
        y_pred = np.dot(self.theta.T, x)
        error = y - y_pred
        
        # 2. Calculate Gain Vector (Kalman Gain)
        # k = (P * x) / (lambda + x^T * P * x)
        Px = np.dot(self.P, x)
        denominator = self.lam + np.dot(x.T, Px)
        k = Px / denominator
        
        # 3. Update Efficiency Matrix (P)
        # P_new = (P - k * x^T * P) / lambda
        term = np.dot(k, np.dot(x.T, self.P))
        self.P = (self.P - term) / self.lam
        
        # 4. Update Coefficients (theta)
        # theta_new = theta + k * error
        self.theta = self.theta + k * error
        
        # Track stats
        self.n_updates += 1
        self.error_history.append(float(error))
        if len(self.error_history) > 1000:
            self.error_history.pop(0)
            
        return float(error)
        
    def get_coefficients(self) -> np.ndarray:
        return self.theta.flatten()
    
    def get_confidence_score(self) -> float:
        """Return a 0-1 score based on recent prediction error stability."""
        if self.n_updates < 10:
            return 0.1
        recent_errors = np.array(self.error_history[-50:])
        mse = np.mean(recent_errors**2)
        # Arbitrary scaling: MSE of 0 is 100% confidence, MSE > 1 is low confidence
        confidence = 1.0 / (1.0 + mse)
        return min(max(confidence, 0.0), 1.0)


class PortfolioOptimizer:
    """
    Vectorized Monte Carlo optimizer for portfolio allocation.
    Calculates Efficient Frontier, Max Sharpe, and Min Variance portfolios.
    """
    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        Args:
            returns_df: DataFrame of asset returns (dates as index).
            risk_free_rate: Annualized risk-free rate for Sharpe calculation.
        """
        self.returns = returns_df
        self.rf = risk_free_rate
        self.assets = returns_df.columns.tolist()
        self.n_assets = len(self.assets)
        self.mean_returns = returns_df.mean() * 12 # Annualized
        self.cov_matrix = returns_df.cov() * 12    # Annualized
        
    def simulate_frontier(self, n_portfolios: int = 5000) -> Dict[str, Any]:
        """
        Simulate random portfolios to generate the Efficient Frontier.
        """
        if self.n_assets < 2:
            return {}
            
        results = np.zeros((3, n_portfolios))
        weights_record = np.zeros((n_portfolios, self.n_assets))
        
        # Vectorized simulation might be tricky with constraints, doing loop for clarity/safety
        # But we can speed it up:
        for i in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            weights_record[i, :] = weights
            
            # Portfolio Return
            p_ret = np.sum(weights * self.mean_returns)
            
            # Portfolio Volatility
            p_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Sharpe Ratio
            p_sharpe = (p_ret - self.rf) / p_vol
            
            results[0,i] = p_ret
            results[1,i] = p_vol
            results[2,i] = p_sharpe
            
        # Identify Key Portfolios
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])
        
        max_sharpe_port = {
            "Return": results[0, max_sharpe_idx],
            "Volatility": results[1, max_sharpe_idx],
            "Sharpe": results[2, max_sharpe_idx],
            "Weights": dict(zip(self.assets, weights_record[max_sharpe_idx]))
        }
        
        min_vol_port = {
            "Return": results[0, min_vol_idx],
            "Volatility": results[1, min_vol_idx],
            "Sharpe": results[2, min_vol_idx],
            "Weights": dict(zip(self.assets, weights_record[min_vol_idx]))
        }
        
        frontier_data = pd.DataFrame({
            "Return": results[0],
            "Volatility": results[1],
            "Sharpe": results[2]
        })
        
        return {
            "max_sharpe": max_sharpe_port,
            "min_variance": min_vol_port,
            "frontier_data": frontier_data
        }
        
    def calculate_rebalancing_trades(
        self, 
        current_holdings: Dict[str, float], 
        target_allocation: Dict[str, float],
        total_capital: float
    ) -> pd.DataFrame:
        """
        Start with Current Holdings ($).
        Target is Allocation % of Total Capital.
        Generate list of Buys/Sells (Difference).
        """
        trades = []
        
        for asset in set(list(current_holdings.keys()) + list(target_allocation.keys())):
            current_val = current_holdings.get(asset, 0.0)
            target_pct = target_allocation.get(asset, 0.0)
            target_val = total_capital * target_pct
            
            diff = target_val - current_val
            
            if abs(diff) < 1.0: # Ignore negligible trades
                continue
                
            trades.append({
                "Asset": asset,
                "Current Value": current_val,
                "Target Value": target_val,
                "Trade Value": diff,
                "Action": "BUY" if diff > 0 else "SELL",
                "Pct Portfolio": target_pct
            })
            
        return pd.DataFrame(trades).sort_values("Trade Value", ascending=False)

    def optimize_max_sharpe_ratio(self) -> Dict[str, Any]:
        """
        Calculate the exact Maximum Sharpe Ratio portfolio using SLSRP/scipy.optimize.
        Constraints: Weights sum to 1, 0 <= w <= 1 (Long Only).
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            LOGGER.warning("Scipy not found. Falling back to Monte Carlo simulation.")
            return self.simulate_frontier(n_portfolios=10000).get("max_sharpe", {})

        n = self.n_assets
        args = (self.mean_returns, self.cov_matrix, self.rf)
        
        # Initial Guess (Equal Weight)
        init_guess = np.array([1.0 / n for _ in range(n)])
        
        # Constraints: Sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: 0 <= weight <= 1
        bounds = tuple((0.0, 1.0) for _ in range(n))
        
        def neg_sharpe(weights, mean_returns, cov_matrix, rf):
            p_ret = np.sum(weights * mean_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - (p_ret - rf) / p_vol
            
        result = minimize(
            neg_sharpe, 
            init_guess, 
            args=args, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if not result.success:
            LOGGER.warning(f"Max Sharpe optimization failed: {result.message}")
            return self.simulate_frontier(n_portfolios=5000).get("max_sharpe", {})
            
        # Extract Results
        opt_weights = result.x
        opt_ret = np.sum(opt_weights * self.mean_returns)
        opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(self.cov_matrix, opt_weights)))
        opt_sharpe = (opt_ret - self.rf) / opt_vol
        
        # Filter clutter (weights < 1%)
        weights_dict = {
            self.assets[i]: w 
            for i, w in enumerate(opt_weights) 
            if w > 0.001
        }
        
        return {
            "Return": opt_ret,
            "Volatility": opt_vol,
            "Sharpe": opt_sharpe,
            "Weights": weights_dict
        }
