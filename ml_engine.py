
"""
Machine Learning & Optimization Engine
--------------------------------------
Provides continuous learning capabilities (Recursive Least Squares) and
portfolio optimization (Monte Carlo Efficient Frontier) using pure Numpy.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Callable

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
        
    def simulate_frontier(self, n_portfolios: int = 5000, progress_callback: Callable[[float], None] = None) -> Dict[str, Any]:
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
            # Report progress every 100 iterations
            if progress_callback and i % 100 == 0:
                progress_callback(i / n_portfolios)

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
        
        if progress_callback:
            progress_callback(1.0)
            
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

    def optimize_max_sharpe_ratio(self, progress_callback: Callable[[float], None] = None) -> Dict[str, Any]:
        """
        Calculate the exact Maximum Sharpe Ratio portfolio using SLSRP/scipy.optimize.
        Constraints: Weights sum to 1, 0 <= w <= 1 (Long Only).
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            LOGGER.warning("Scipy not found. Falling back to Monte Carlo simulation.")
            return self.simulate_frontier(n_portfolios=10000, progress_callback=progress_callback).get("max_sharpe", {})

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
            # Fallback to Equal Weight (Fast) instead of Monte Carlo (Slow)
            # LOGGER.warning(f"Max Sharpe optimization failed: {result.message}. Using Equal Weight.")
            equal_weight = 1.0 / n if n > 0 else 0
            weights_dict = {a: equal_weight for a in self.assets}
            
            # Recalculate metrics for EW
            ew_ret = np.sum(init_guess * self.mean_returns)
            ew_vol = np.sqrt(np.dot(init_guess.T, np.dot(self.cov_matrix, init_guess)))
            ew_sharpe = (ew_ret - self.rf) / ew_vol if ew_vol > 1e-6 else 0.0
            
            return {
                "Return": ew_ret,
                "Volatility": ew_vol,
                "Sharpe": ew_sharpe,
                "Weights": weights_dict
            }
            
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


class PredictiveAlphaEngine:
    """
    Manages a universe of OnlineLinearRegressors to predict returns for multiple assets.
    """
    def __init__(self, assets: List[str], lambda_factor: float = 0.98):
        self.assets = assets
        self.models = {
            asset: OnlineLinearRegressor(n_features=3, lambda_factor=lambda_factor) 
            for asset in assets
        }
        self.feature_history = {asset: [] for asset in assets}
        
    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract features from a price history window.
        Features: 
        1. 1-Day Return
        2. 5-Day Return
        3. 10-Day Volatility (approx)
        Requires at least 11 days of history.
        """
        if len(prices) < 11:
            return np.zeros(3)
            
        p_now = prices[-1]
        p_prev = prices[-2]
        p_5d = prices[-6]
        
        # 1. Momentum (1D)
        ret_1d = (p_now - p_prev) / p_prev if p_prev > 1e-6 else 0.0
        
        # 2. Momentum (5D)
        ret_5d = (p_now - p_5d) / p_5d if p_5d > 1e-6 else 0.0
        
        # 3. Volatility (10D std dev of returns)
        recent_prices = prices[-11:]
        # Safe division
        denom = recent_prices[:-1]
        # Replace zeros with nan to avoid warning, then handle
        with np.errstate(divide='ignore', invalid='ignore'):
            recent_rets = np.diff(recent_prices) / np.where(denom == 0, np.nan, denom)
        
        recent_rets = np.nan_to_num(recent_rets, nan=0.0)
        vol_10d = np.std(recent_rets)
        
        return np.array([ret_1d, ret_5d, vol_10d])
        
    def update(self, market_snapshot: pd.DataFrame):
        """
        Update all models with the latest market data row.
        market_snapshot: A single row DataFrame (or Series) with columns as Assets.
        """
        # We need history to generate features.
        # This function updates the history AND trains the model on the *previous* prediction.
        
        # For simplicity in this streaming implementation:
        # 1. We assume 'market_snapshot' contains the CLOSE price for today.
        # 2. We calculate today's return (Target for YESTERDAY's prediction).
        # 3. We extract YESTERDAY's features.
        # 4. We update the model.
        # 5. We then extract TODAY's features to use for TOMORROW's prediction.
        pass # The logic is handled better in batch within Backtester for now.


class Backtester:
    """
    Simulates the "Always Bought and Sold" strategy using the PredictiveAlphaEngine.
    """
    def __init__(self, initial_capital: float = 10000.0, transaction_cost_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.cost_pct = transaction_cost_pct
        
    def run(self, price_data: pd.DataFrame, window_size: int = 30, progress_callback: Callable[[float], None] = None) -> Dict[str, Any]:
        """
        Run the simulation.
        price_data: DataFrame with Date index and Asset columns (Prices).
        window_size: Warm-up period for features.
        """
        assets = price_data.columns.tolist()
        ml_engine = PredictiveAlphaEngine(assets)
        optimizer = PortfolioOptimizer(pd.DataFrame(), risk_free_rate=0.04) # Placeholder init
        
        # Storage
        equity_curve = [self.initial_capital]
        allocations = []
        date_index = price_data.index
        
        current_capital = self.initial_capital
        current_weights = {a: 0.0 for a in assets}
        
        # We step through the data day by day
        # Start after window_size
        n_rows = len(price_data)
        if n_rows < window_size + 1:
            return {"error": "Not enough data"}
            
        progress_steps = []
        
        # Pre-calculate returns for efficiency
        returns_df = price_data.pct_change().fillna(0)
        
        total_steps = n_rows - 1 - window_size
        
        for idx, t in enumerate(range(window_size, n_rows - 1)):
            if progress_callback and idx % 5 == 0:
                 progress_callback(idx / total_steps)
            # 0. Data Slicing
            # Knowledge Cutoff: t (Today) taking action for t+1 (Tomorrow)
            # We treat 't' as "Today's Close". we rebalance at Close or Open of next? 
            # Let's assume we rebalance at T using Close prices.
            
            today_date = date_index[t]
            history_prices = price_data.iloc[:t+1]
            
            # 1. Update Models (Train)
            # Target: Return from t-1 to t
            # Features: Calculated at t-1
            if t > window_size:
                # We can learn from the transition (t-1) -> (t)
                target_returns = returns_df.iloc[t] # Return of "Today"
                
                # We need features from t-1
                prev_prices_window = price_data.iloc[:t] # Up to yesterday
                
                for asset in assets:
                    # Extract features as they would have looked yesterday
                    # This implies getting a window ending at t-1
                    # Optimization: In a real loop, we'd cache features. 
                    # Here we re-extract for clarity (slower but safer).
                    
                    # Window ensuring we have 11 days ending at t-1
                    # t is index. t-1 is yesterday.
                    # slice iloc [t-11 : t] gives 11 items ending at t-1 (exclusive of t?)
                    # No, iloc[:t] excludes t. So it gives 0..t-1.
                    
                    hist_slice = price_data[asset].values[max(0, t-12) : t] 
                    if len(hist_slice) > 10:
                        feats = ml_engine._extract_features(hist_slice)
                        actual_ret = target_returns[asset]
                        
                        # Train
                        ml_engine.models[asset].update(feats, actual_ret)
            
            # 2. Predict Next Day (t+1) Alpha
            # We use features from "Today" (t)
            predicted_returns = {}
            for asset in assets:
                curr_slice = price_data[asset].values[max(0, t-11) : t+1]
                feats = ml_engine._extract_features(curr_slice)
                pred_ret = ml_engine.models[asset].predict(feats)
                predicted_returns[asset] = pred_ret
            
            # 3. Optimize Portfolio
            # We construct a synthetic "Expected Returns" vector for the optimizer
            # And use recent Covariance (e.g., last 30 days)
            
            recent_returns = returns_df.iloc[t-30:t+1]
            cov = recent_returns.cov() * 252 # Annualized
            
            # Manually inject predictions into Optimizer logic? 
            # The existing Optimizer calculates mean_returns from history.
            # We want to OVERRIDE that with our Predictions.
            
            # Helper: Solve for weights given (Mu, Sigma)
            # We'll use a simplified sharpe optimizer here or reuse the class
            # Let's add a method to PortfolioOptimizer to accept custom Mu
            
            # Hack: Create a dummy optimizer instance just for the math
            opt = PortfolioOptimizer(recent_returns) 
            # Override internal mean_returns with our predictions (Annualized)
            pred_mu_series = pd.Series(predicted_returns) * 252 
            
            # Safety: Clip extreme predictions to avoid solver errors
            pred_mu_series = pred_mu_series.clip(-5.0, 5.0)
            
            opt.mean_returns = pred_mu_series
            opt.cov_matrix = cov
            
            # Optimize!
            result = opt.optimize_max_sharpe_ratio()
            target_weights = result.get("Weights", {})
            
            # 4. Execute & Record
            # We assume we get the return of t+1
            next_day_ret = returns_df.iloc[t+1]
            
            # Calculate Portfolio Return
            # P_ret = sum(weight * asset_return)
            # Transaction costs: sum(abs(w_new - w_old)) * cost
            
            port_ret = 0.0
            turnover = 0.0
            
            for asset in assets:
                w_new = target_weights.get(asset, 0.0)
                w_old = current_weights.get(asset, 0.0)
                r = next_day_ret[asset]
                
                port_ret += w_new * r
                turnover += abs(w_new - w_old)
                
                current_weights[asset] = w_new # Update for next step
            
            # Apply costs
            cost = turnover * self.cost_pct
            net_ret = port_ret - cost
            
            current_capital *= (1.0 + net_ret)
            equity_curve.append(current_capital)
            allocations.append(current_weights.copy())
            
        # Compile Results
        equity_series = pd.Series(equity_curve, index=date_index[window_size:])
        
        # Metrics
        total_ret = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        cagr = (equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1
        daily_rets = equity_series.pct_change().dropna()
        sharpe = (daily_rets.mean() / daily_rets.std()) * (252**0.5) if daily_rets.std() > 0 else 0
        
        return {
            "equity_curve": equity_series,
            "final_capital": current_capital,
            "metrics": {
                "Total Return": total_ret,
                "CAGR": cagr,
                "Sharpe": sharpe,
                "Volatility": daily_rets.std() * (252**0.5)
            },
            "final_weights": current_weights,
            "allocations": allocations
        }
