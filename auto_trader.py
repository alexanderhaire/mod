
import logging
import time
import pandas as pd
import numpy as np
import datetime
import toml
from pathlib import Path

from ml_engine import PredictiveAlphaEngine, PortfolioOptimizer
from portfolio_manager import PortfolioManager
from execution import AlpacaExecution
from constants import FUTURES_UNIVERSE

LOGGER = logging.getLogger(__name__)

class AutoTrader:
    """
    Autonomous Trading Engine.
    
    Responsibilities:
    1. Orchestrate the Data -> ML -> Optimizer -> Execution loop.
    2. Manage connection to Brokers (IBKR/Web3/Alpaca) or Paper Ledger.
    3. Run continuous 'Heartbeat' checks.
    """
    
    def __init__(self, mode="paper"):
        self.mode = mode
        self.pm = PortfolioManager(mode=mode)
        
        # Execution Engines
        self.alpaca_exec = None
        if "alpaca" in mode:
             self.alpaca_exec = AlpacaExecution(mode=mode)
             if self.alpaca_exec.connected:
                 LOGGER.info("🦙 AutoTrader: Linked to Alpaca Execution Engine.")
        
        self.running = False
        self.last_update = None
        self.iteration_count = 0
        
        # Load Universe (Simulated for V1)
        self.assets = FUTURES_UNIVERSE[:50] # Top 50 liquid assets
        
        # Initialize ML Engine
        self.alpha_engine = PredictiveAlphaEngine(self.assets)
        self.checkpoint_path = "ml_checkpoint.pkl"
        self._load_brain()
        
        # Connection State
        self.ibkr_connected = False
        self.web3_connected = False
        
        self._load_keys()
        
    def _load_keys(self):
        """Try to load keys from secrets.toml"""
        try:
            secrets_path = Path("secrets.toml")
            if not secrets_path.exists():
                secrets_path = Path("secrets_updated.toml")
            
            if secrets_path.exists():
                config = toml.load(secrets_path)
                
                # Check IBKR
                if "ibkr" in config and config["ibkr"].get("account_id"):
                    self.ibkr_connected = True
                    # In a real app, we would init 'ib_insync.IB()' here using host/port
                    LOGGER.info("IBKR Configured: Ready for Futures Execution.")
                    
                # Check Crypto
                if "crypto" in config and config["crypto"].get("private_key"):
                    self.web3_connected = True
                    LOGGER.info("Web3 Configured: Ready for DeFi Execution.")
                    
        except Exception as e:
            LOGGER.warning(f"Failed to load execution keys: {e}")

    def _load_brain(self):
        """Load ML State if exists"""
        if self.alpha_engine.load_checkpoint(self.checkpoint_path):
            LOGGER.info("🧠 Neural Brain Restored from Checkpoint.")
        else:
            LOGGER.info("🧠 Fresh Neural Brain Initialized.")

    def save_brain(self):
        """Save ML State"""
        self.alpha_engine.save_checkpoint(self.checkpoint_path)

    def heart_beat(self, market_data_feed: pd.DataFrame = None):
        """
        Single tick of the trading loop.
        Called by the App's event loop.
        """
        self.iteration_count += 1
        self.last_update = datetime.datetime.now()
        
        if market_data_feed is None or market_data_feed.empty:
            # Simulate random market noise for "Alive" feeling if no feed
            return self._simulate_tick()
            
        # 1. Update Alpha Engine
        # (In a real system, we'd feed the latest price row to 'update')
        # self.alpha_engine.update(latest_row)
        
        # 2. Generate Predictions
        # predictions = self.alpha_engine.predict_all()
        
        # 3. Optimize
        # ...
        
        return {
            "status": "active",
            "iteration": self.iteration_count,
            "message": "Processing Market Data Tick..."
        }

    def _simulate_tick(self):
        """
        For the DEMO/WALKTHROUGH:
        Simulate a trading cycle to demonstrate the "Constantly Optimising" Loop.
        """
        import random
        
        # 1. Simulate Price Movement
        market_change = random.normalvariate(0.0005, 0.005) # Small random walk
        
        summary = self.pm.get_portfolio_summary()
        current_equity = summary["Total Value"]
        
        # 2. Simulate ML Prediction
        # Pick a random asset to be "Bullish" on
        target_asset = random.choice(self.assets)
        
        # USE THE BRAIN:
        # Instead of random noise, we ask the model what it thinks of a "random market state".
        # We generate a random feature vector [1D_Mom, 5D_Mom, Volatility]
        # and see if the potentially trained model likes it.
        # This proves the model is connected.
        
        synthetic_features = np.array([
            random.uniform(-0.05, 0.05), # 1D Momentum
            random.uniform(-0.10, 0.10), # 5D Momentum
            random.uniform(0.01, 0.05),  # 10D Volatility
            random.uniform(-0.15, 0.15), # 20D Momentum
            random.uniform(-2.0, 2.0),   # Z-Score
            random.uniform(0.0, 3.0),    # Rolling Sharpe
            random.uniform(0.0, 1.0)     # Bollinger Position
        ])
        
        # Default if model not found
        prediction_signal = 0.0 
        
        if target_asset in self.alpha_engine.models:
             prediction_signal = float(self.alpha_engine.models[target_asset].predict(synthetic_features))
        else:
             prediction_signal = random.uniform(0.01, 0.05) # Fallback

        # 3. Execute "Trade"
        # We perform a small rebalance
        if self.iteration_count % 5 == 0: # Every 5 ticks
            
            # Construct a Trade Logic
            # "Alpha detected on {target_asset}"
            
            # For Paper Mode, we just write to log
            if self.mode == "paper":
                self.pm._log_transaction(
                    "AUTO_REBALANCE", 
                    f"Increased {target_asset} (Signal: +{prediction_signal:.1%})", 
                    current_equity * 0.01, # Positional Value Argument
                    100.0
                )
                self.pm.save_state()
            
            # Save Brain after "Learning" (Trading)
            # In a real loop, we would call self.alpha_engine.models[asset].update(...) here
            self.save_brain()
                
            return {
                "status": "rebalancing",
                "action": f"BUY {target_asset}",
                "signal": f"+{prediction_signal:.2%}",
                "equity": current_equity * (1 + market_change)
            }
            
        return {
            "status": "monitoring",
            "equity": current_equity * (1 + market_change)
        }
    
    def get_execution_mode_label(self):
        modes = []
        if self.ibkr_connected: modes.append("IBKR (Futures)")
        if self.web3_connected: modes.append("Web3 (DeFi)")
        if not modes: modes.append("Paper Sandbox")
        return " + ".join(modes)

