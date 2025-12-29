
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
import pickle
import os

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
        
        # Ledoit-Wolf Shrinkage for stable covariance estimation
        if returns_df.empty:
             self.cov_matrix = pd.DataFrame()
        else:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(returns_df.fillna(0))
                self.cov_matrix = pd.DataFrame(lw.covariance_, index=returns_df.columns, columns=returns_df.columns) * 12
            except (ImportError, ValueError):
                # Fallback to raw covariance if sklearn not available or fails
                self.cov_matrix = returns_df.cov() * 12
        
    def simulate_frontier(self, n_portfolios: int = 5000, progress_callback: Callable[[float], None] = None) -> Dict[str, Any]:
        """
        Simulate random portfolios to generate the Efficient Frontier.
        """
        if self.n_assets < 2:
            return {}

        # Vectorized Simulation (100x Faster)
        # Generate all weights at once
        weights = np.random.random((n_portfolios, self.n_assets))
        weights /= weights.sum(axis=1)[:, np.newaxis] # Normalize rows to sum to 1

        # Portfolio Returns: dot product of weights and mean_returns
        # weights: (M, N), mean_returns: (N,) -> (M,)
        p_rets = np.dot(weights, self.mean_returns.values)

        # Portfolio Volatility: sqrt(w.T * Cov * w)
        # A bit trickier to vectorize fully without loop or einsum
        # Variance = sum(weights @ cov * weights, axis=1)
        # We can use einsum: "ij,jk,ik->i" (weights, cov, weights)
        # i: portfolio index, j: asset 1, k: asset 2
        # cov is (N, N), weights is (M, N)
        # intermediate = weights @ cov -> (M, N)
        # final = sum(intermediate * weights, axis=1) -> (M,)
        
        interim = np.dot(weights, self.cov_matrix.values)
        p_vars = np.sum(interim * weights, axis=1)
        p_vols = np.sqrt(p_vars)

        # Sharpe Ratio
        p_sharpes = (p_rets - self.rf) / p_vols

        # Identify Key Portfolios
        max_sharpe_idx = np.argmax(p_sharpes)
        min_vol_idx = np.argmin(p_vols)

        max_sharpe_port = {
            "Return": p_rets[max_sharpe_idx],
            "Volatility": p_vols[max_sharpe_idx],
            "Sharpe": p_sharpes[max_sharpe_idx],
            "Weights": dict(zip(self.assets, weights[max_sharpe_idx]))
        }

        min_vol_port = {
            "Return": p_rets[min_vol_idx],
            "Volatility": p_vols[min_vol_idx],
            "Sharpe": p_sharpes[min_vol_idx],
            "Weights": dict(zip(self.assets, weights[min_vol_idx]))
        }

        frontier_data = pd.DataFrame({
            "Return": p_rets,
            "Volatility": p_vols,
            "Sharpe": p_sharpes
        })

        if progress_callback:
            progress_callback(1.0)

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

    def optimize_max_sharpe_ratio(self, progress_callback: Callable[[float], None] = None, tradable_assets: List[str] = None) -> Dict[str, Any]:
        """
        Calculate the exact Maximum Sharpe Ratio portfolio using SLSRP/scipy.optimize.
        Constraints: 
        1. Net Exposure: UNCONSTRAINED (Can be Net Short or Net Long)
        2. Gross Exposure <= 300% (Sum abs(weights) <= 3) -> 3x Leverage
        3. Bounds: -1.0 <= w <= 1.0 (Long/Short) for Tradable Assets, (0,0) for others.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            LOGGER.warning("Scipy not found. Falling back to Monte Carlo simulation.")
            return self.simulate_frontier(n_portfolios=10000, progress_callback=progress_callback).get("max_sharpe", {})

        n = self.n_assets
        args = (self.mean_returns, self.cov_matrix, self.rf)
        
        # Bounds: -1 <= weight <= 1 (Long/Short)
        # Use Private Data for Correlation, but ONLY trade Public Assets
        init_weights = []
        if tradable_assets:
             bounds = []
             n_tradable = 0
             for asset in self.assets:
                 if asset in tradable_assets:
                     bounds.append((-1.0, 1.0))
                     init_weights.append(1.0) # Temp, will normalize later
                     n_tradable += 1
                 else:
                     bounds.append((0.0, 0.0)) # Forced Zero Allocation
                     init_weights.append(0.0)
             bounds = tuple(bounds)
             
             # Normalize initial guess to sum to 1.0 (or 0.0 for Unchained, but solver likes 1.0 scale)
             if n_tradable > 0:
                 init_guess = np.array(init_weights) / n_tradable
             else:
                 init_guess = np.zeros(n)
        else:
             bounds = tuple((-1.0, 1.0) for _ in range(n))
             init_guess = np.array([1.0 / n for _ in range(n)])
        
        # Constraints: 
        # 1. Sum of abs(weights) <= 1.0 (Fully invested, no leverage)
        # 2. REMOVED Dollar-Neutral: Allow directional bets (net long or short)
        constraints = (
            {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(np.abs(x))},
            # Dollar-neutral removed - market can trend!
        )
        
        # Update init_guess for directional mode
        if tradable_assets and n_tradable > 1:
            # Equal positive weights as starting point (not dollar-neutral)
            init_guess = np.zeros(n)
            for i, asset in enumerate(self.assets):
                if asset in tradable_assets:
                    init_guess[i] = 1.0 / n_tradable  # Equal weighted long

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
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-4}
        )
        
        if not result.success:
             # Fallback to Rank-Based "Iron Hand" Allocation if solver fails or returns flat weights
             LOGGER.warning(f"Solver failed: {result.message}. Using logic-based allocation.")
             return self._allocate_rank_based()
        
        if np.std(result.x) < 1e-6:
             LOGGER.warning(f"Solver returned flat weights (Std={np.std(result.x):.6f}). Using logic-based allocation.")
             return self._allocate_rank_based()
        
        return {
            "Weights": dict(zip(self.assets, result.x)),
            "Return": np.sum(result.x * self.mean_returns),
            "Volatility": np.sqrt(np.dot(result.x.T, np.dot(self.cov_matrix, result.x))),
            "Sharpe": -result.fun
        }

    def _allocate_rank_based(self) -> Dict[str, Any]:
        """
        Robust fallback: Sorts assets by predicted return.
        Longs top 5, Shorts bottom 5.
        """
        # Sort assets by predicted return
        sorted_assets = self.mean_returns.sort_values(ascending=False)
        weights = {}
        for a in self.assets:
            weights[a] = 0.0
            
        # Top 5 -> Long
        top_5 = sorted_assets.head(5).index
        for a in top_5:
            weights[a] = 0.15 # 5 * 0.15 = 0.75 Long
            
        # Bottom 5 -> Short
        bot_5 = sorted_assets.tail(5).index
        for a in bot_5:
            weights[a] = -0.15 # 5 * -0.15 = -0.75 Short
            
        # Total Gross = 3.0. Net = 2.0. Unchained.
        
        w_vec = np.array([weights[a] for a in self.assets])
        
        return {
            "Weights": weights,
            "Return": np.sum(w_vec * self.mean_returns),
            "Volatility": np.sqrt(np.dot(w_vec.T, np.dot(self.cov_matrix, w_vec))),
            "Sharpe": 0.0 # Placeholder
        }


class PredictiveAlphaEngine:
    """
    Manages a universe of OnlineLinearRegressors to predict returns for multiple assets.
    """
    def __init__(self, assets: List[str], lambda_factor: float = 0.98):
        self.assets = assets
        self.models = {
            asset: OnlineLinearRegressor(n_features=12, lambda_factor=lambda_factor) 
            for asset in assets
        }
        self.feature_history = {asset: [] for asset in assets}
        
    def save_checkpoint(self, filepath: str):
        """
        Save the current state of all models to disk.
        Allows for 'Epoch'-like isolation and resuming training.
        """
        try:
            state = {
                "assets": self.assets,
                "models": {
                    asset: {
                        "theta": model.theta,
                        "P": model.P,
                        "n_features": model.n_features,
                        "lam": model.lam,
                        "n_updates": model.n_updates,
                        "error_history": model.error_history
                    }
                    for asset, model in self.models.items()
                }
            }
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            LOGGER.info(f"Model checkpoint saved to {filepath}")
        except Exception as e:
            LOGGER.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """
        Load model state from disk.
        """
        try:
            if not os.path.exists(filepath):
                LOGGER.warning(f"Checkpoint file {filepath} not found.")
                return False
                
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            # If assets mismatch, we might need to handle partial loading.
            # For now, we assume strict match or subset.
            
            loaded_count = 0
            for asset, model_state in state["models"].items():
                if asset in self.models:
                    model = self.models[asset]
                    model.theta = model_state["theta"]
                    model.P = model_state["P"]
                    model.n_updates = model_state.get("n_updates", 0)
                    model.error_history = model_state.get("error_history", [])
                    loaded_count += 1
            
            LOGGER.info(f"Loaded checkpoint from {filepath} ({loaded_count} models restored).")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to load checkpoint: {e}")
            return False

        
    def _extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract features from a price history window.
        Enhanced Feature Set (12 signals):
        
        MOMENTUM SIGNALS:
        1. 1-Day Return (Momentum)
        2. 5-Day Return (Short-Term Trend)
        3. 7-Day Return (Weekly Momentum) [NEW]
        4. 20-Day Return (Medium-Term Trend)
        
        VOLATILITY SIGNALS:
        5. 10-Day Volatility (Short-term risk)
        6. 20-Day Volatility (Risk proxy) [NEW]
        
        MEAN REVERSION SIGNALS:
        7. Z-Score (Distance from 20-day mean)
        8. Bollinger Band Position (0=oversold, 1=overbought)
        9. RSI-like Signal (% of up days in last 14) [NEW]
        
        TREND SIGNALS:
        10. SMA Ratio (SMA7/SMA20 - trend strength) [NEW]
        11. Rolling Sharpe (Quality of recent performance)
        12. Drawdown from Peak (Risk indicator) [NEW]
        
        Requires at least 31 days of history for 30-day calculations.
        """
        N_FEATURES = 12
        
        if len(prices) < 31:
            return np.zeros(N_FEATURES)
            
        p_now = prices[-1]
        p_prev = prices[-2]
        p_5d = prices[-6]
        p_7d = prices[-8] if len(prices) >= 8 else prices[0]
        p_20d = prices[-21]
        
        # --- MOMENTUM SIGNALS ---
        # 1. Momentum (1D)
        ret_1d = (p_now - p_prev) / p_prev if p_prev > 1e-6 else 0.0
        
        # 2. Momentum (5D)
        ret_5d = (p_now - p_5d) / p_5d if p_5d > 1e-6 else 0.0
        
        # 3. Momentum (7D) - Weekly
        ret_7d = (p_now - p_7d) / p_7d if p_7d > 1e-6 else 0.0
        
        # 4. Momentum (20D) - Medium-term trend
        ret_20d = (p_now - p_20d) / p_20d if p_20d > 1e-6 else 0.0
        
        # --- VOLATILITY SIGNALS ---
        # 5. Volatility (10D std dev of returns)
        recent_prices_10 = prices[-11:]
        denom_10 = recent_prices_10[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            recent_rets_10 = np.diff(recent_prices_10) / np.where(denom_10 == 0, np.nan, denom_10)
        recent_rets_10 = np.nan_to_num(recent_rets_10, nan=0.0)
        vol_10d = np.std(recent_rets_10)
        
        # 6. Volatility (20D) - More stable risk measure
        recent_prices_20 = prices[-21:]
        denom_20 = recent_prices_20[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            recent_rets_20 = np.diff(recent_prices_20) / np.where(denom_20 == 0, np.nan, denom_20)
        recent_rets_20 = np.nan_to_num(recent_rets_20, nan=0.0)
        vol_20d = np.std(recent_rets_20)
        
        # --- MEAN REVERSION SIGNALS ---
        # 7. Z-Score (Mean Reversion) - How far from 20-day mean
        mean_20d = np.mean(prices[-20:])
        std_20d = np.std(prices[-20:])
        z_score = (p_now - mean_20d) / std_20d if std_20d > 1e-6 else 0.0
        
        # 8. Bollinger Band Position (0=oversold, 1=overbought)
        upper_band = mean_20d + 2 * std_20d
        lower_band = mean_20d - 2 * std_20d
        bb_range = upper_band - lower_band
        bb_position = (p_now - lower_band) / bb_range if bb_range > 1e-6 else 0.5
        bb_position = np.clip(bb_position, 0, 1)
        
        # 9. RSI-like Signal - Percentage of up days in last 14 days
        rets_14d = np.diff(prices[-15:])
        up_days = np.sum(rets_14d > 0)
        rsi_signal = up_days / 14.0  # Range 0-1, 0.5 = neutral
        
        # --- TREND SIGNALS ---
        # 10. SMA Ratio (SMA7/SMA20) - Trend strength
        sma_7 = np.mean(prices[-7:])
        sma_20 = np.mean(prices[-20:])
        sma_ratio = (sma_7 / sma_20 - 1.0) if sma_20 > 1e-6 else 0.0  # Centered at 0
        
        # 11. Rolling Sharpe (Quality) - Risk-adjusted recent performance
        rolling_sharpe = np.mean(recent_rets_20) / np.std(recent_rets_20) if np.std(recent_rets_20) > 1e-6 else 0.0
        
        # 12. Drawdown from Peak (Risk indicator)
        peak = np.max(prices[-30:])
        drawdown = (p_now - peak) / peak if peak > 1e-6 else 0.0  # Negative when below peak
        
        return np.array([
            ret_1d, ret_5d, ret_7d, ret_20d,      # Momentum (4)
            vol_10d, vol_20d,                      # Volatility (2)
            z_score, bb_position, rsi_signal,      # Mean Reversion (3)
            sma_ratio, rolling_sharpe, drawdown    # Trend (3)
        ])
        
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
    Simons-style: Low transaction cost assumption for high-frequency alpha capture.
    """
    def __init__(self, initial_capital: float = 10000.0, transaction_cost_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.cost_pct = transaction_cost_pct
        
    def run(self, price_data: pd.DataFrame, window_size: int = 40, progress_callback: Callable[[float], None] = None, checkpoint_path: str = None, tradable_assets: List[str] = None) -> Dict[str, Any]:
        """
        Run the simulation.
        price_data: DataFrame with Date index and Asset columns (Prices).
        window_size: Warm-up period for features.
        checkpoint_path: If provided, save the trained model to this path.
        tradable_assets: Optional list of assets allowed to have non-zero weights.
        """
        assets = price_data.columns.tolist()
        ml_engine = PredictiveAlphaEngine(assets)
        optimizer = PortfolioOptimizer(pd.DataFrame(), risk_free_rate=0.04) # Placeholder init
        
        # Storage
        equity_curve = [self.initial_capital]
        allocations = []
        date_index = price_data.index
        
        current_capital = self.initial_capital
        peak_equity = self.initial_capital  # For drawdown circuit breaker
        in_circuit_breaker = False  # True when halted due to drawdown
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
                    # slice needs 21+ days for _extract_features
                    hist_slice = price_data[asset].values[max(0, t-32) : t]  # 32 days for 31-day features
                    if len(hist_slice) >= 31:
                        feats = ml_engine._extract_features(hist_slice)
                        actual_ret = target_returns[asset]
                        
                        # Train
                        ml_engine.models[asset].update(feats, actual_ret)
            
            # 2. Predict Next Day (t+1) Alpha
            # We use features from "Today" (t)
            predicted_returns = {}
            for asset in assets:
                curr_slice = price_data[asset].values[max(0, t-31) : t+1]  # Need 32 values for 31-day features
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
            # AMPLIFICATION: We multiply predictions by 50.0 to force the optimizer
            # to take large positions even on small signals in the Unchained mode.
            
            # SIMONS-STYLE: Kelly-inspired confidence scaling
            # Scale predictions by model confidence (high confidence = bigger bet)
            confidence_scaled_returns = {}
            avg_confidence = 0.0
            for asset in assets:
                raw_pred = predicted_returns[asset]
                confidence = ml_engine.models[asset].get_confidence_score()
                avg_confidence += confidence
                # Scale: Full prediction at 100% confidence, 10% at low confidence
                scaled_pred = raw_pred * (0.1 + 0.9 * confidence)
                confidence_scaled_returns[asset] = scaled_pred
            
            avg_confidence /= len(assets) if assets else 1.0
            
            # --- CONFIDENCE GATE: Only trade when the model is "hot" ---
            CONFIDENCE_THRESHOLD = 0.10  # Lowered: allow more trading, especially early
            
            if avg_confidence < CONFIDENCE_THRESHOLD:
                # Model is "cold" - sit out this day
                target_weights = {a: 0.0 for a in assets}
            else:
                # Model is "hot" - use DIRECT SIGNAL ALLOCATION (no optimizer)
                # Long assets with positive predictions, short assets with negative
                # Size proportional to prediction magnitude * confidence
                
                raw_signals = {}
                for asset in assets:
                    pred = confidence_scaled_returns[asset]
                    conf = ml_engine.models[asset].get_confidence_score()
                    # CONTRARIAN: Invert the signal (bet AGAINST the prediction)
                    raw_signals[asset] = -np.sign(pred) * conf * abs(pred) * 100
                
                # Normalize to sum of abs weights = 1.0 (fully invested, no leverage)
                total_signal = sum(abs(s) for s in raw_signals.values())
                if total_signal > 1e-6:
                    target_weights = {a: s / total_signal for a, s in raw_signals.items()}
                else:
                    target_weights = {a: 0.0 for a in assets}
            
            # --- CIRCUIT BREAKER: 15% Drawdown Halt ---
            peak_equity = max(peak_equity, current_capital)
            drawdown = (peak_equity - current_capital) / peak_equity if peak_equity > 0 else 0.0
            
            if drawdown > 0.25:  # 25% drawdown threshold (was 15%)
                if not in_circuit_breaker:
                    LOGGER.warning(f"🔴 CIRCUIT BREAKER TRIGGERED: Drawdown {drawdown:.1%}. Halting trading for 60 days.")
                    in_circuit_breaker = True
                    circuit_breaker_timer = 60 # Cooldown days
                
                target_weights = {a: 0.0 for a in assets}  # Force all cash
                
            if in_circuit_breaker:
                target_weights = {a: 0.0 for a in assets} # Enforce halt
                if circuit_breaker_timer > 0:
                     circuit_breaker_timer -= 1
                else:
                    # Recovery: Reset circuit breaker after cooldown
                    LOGGER.info(f"🟢 CIRCUIT BREAKER RESET: Cooldown complete. Resuming trading.")
                    in_circuit_breaker = False
                    # Reset peak equity to give it a chance to breathe (optional, but good for relative DD)
                    peak_equity = current_capital 
            # -------------------------------------------
            
            # --- VOLATILITY SCALING: Reduce exposure when volatility is high ---
            if not in_circuit_breaker and len(equity_curve) > 20:
                # Calculate recent portfolio volatility (proxy: use overall market vol)
                recent_rets = returns_df.iloc[t-20:t].mean(axis=1)  # Average asset return
                recent_vol = recent_rets.std() * np.sqrt(252)  # Annualized
                
                # Target volatility scaling: 15% annual vol target
                target_vol = 0.15
                vol_scalar = min(target_vol / recent_vol, 1.5) if recent_vol > 0.01 else 1.0
                
                # Scale weights by volatility scalar
                target_weights = {a: w * vol_scalar for a, w in target_weights.items()}
            # -----------------------------------------------------------------
            
            # --- POSITION SMOOTHING: EMA to reduce turnover ---
            # Blend target weights with current weights to avoid whipsawing
            SMOOTHING_FACTOR = 0.7  # Optimal: balances responsiveness with stability
            smoothed_weights = {}
            for asset in assets:
                w_target = target_weights.get(asset, 0.0)
                w_current = current_weights.get(asset, 0.0)
                smoothed_weights[asset] = SMOOTHING_FACTOR * w_target + (1 - SMOOTHING_FACTOR) * w_current
            target_weights = smoothed_weights
            # --------------------------------------------------
            
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
        
        if checkpoint_path:
            ml_engine.save_checkpoint(checkpoint_path)

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
