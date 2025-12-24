import json
import os
import pandas as pd
import datetime
import logging

LOGGER = logging.getLogger(__name__)

PORTFOLIO_FILE = "portfolio_state.json"

class PortfolioManager:
    """
    Manages a persistent portfolio using a local JSON file.
    Tracks: Cash, Holdings, Transaction History.
    """
    def __init__(self, mode="paper"):
        self.mode = mode
        if self.mode == "real":
            self.filepath = "portfolio_state_real.json"
        else:
            self.filepath = "portfolio_state.json"  # Default/Paper
            
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                LOGGER.error(f"Failed to load portfolio: {e}")
        
        # Default State
        return {
            "cash": 0.0,
            "holdings": {},  # Ticker -> Quantity (or Value, simpler to track Value for now as prices change)
            # Tracking Value directly is tricky without price feeds. 
            # We will track "Allocated Capital" (Cost Basis) for simplicity in this V1 Paper Trader.
            "history": []
        }

    def save_state(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Failed to save portfolio: {e}")

    def get_portfolio_summary(self):
        """Returns current breakdown of Cash vs Invested."""
        invested = sum(self.state["holdings"].values())
        total = self.state["cash"] + invested
        return {
            "Cash": self.state["cash"],
            "Invested": invested,
            "Total Value": total,
            "Holdings": self.state["holdings"]
        }

    def deposit_capital(self, amount: float):
        if amount > 0:
            self.state["cash"] += amount
            self._log_transaction("DEPOSIT", "CASH", amount, 1.0)
            self.save_state()
            return True
        return False

    def execute_rebalancing(self, blueprint_df: pd.DataFrame):
        """
        Adjusts portfolio to match the Target Allocation ($) in blueprint_df.
        blueprint_df must have ['Item', 'Allocated Capital'].
        """
        # 1. Liquidate everything first? (Simpler for "Pure Allocator" which assumes clean slate)
        # OR Calculate difference. 
        # For V1, "Pure Allocator" means "This IS your portfolio". 
        # So we sell what's not in target, buy what is.
        
        current_holdings = self.state["holdings"]
        
        # Target State
        target_map = dict(zip(blueprint_df['Item'], blueprint_df['Allocated Capital']))
        
        # 1. Sell / Reduce Overweight
        # (For this Paper Trader, we assume instant execution at current price, 
        # so we just shift capital buckets).
        
        total_needed = sum(target_map.values())
        
        # Check if we have enough total equity
        summary = self.get_portfolio_summary()
        if total_needed > summary["Total Value"] * 1.01: # 1% buffer for rounding
             LOGGER.warning("Insufficient funds for target allocation.")
             # Limit execution to available funds?
             # For now, let's assume the blueprint was generated based on the User's defined Budget.
             # If "Budget" > "Actual Funds", we implicitly "Deposit" the difference or fail?
             # Let's auto-scale the blueprint to fit actual Total Value.
             scale_factor = summary["Total Value"] / total_needed
             target_map = {k: v * scale_factor for k, v in target_map.items()}
        
        self.state["holdings"] = target_map
        self.state["cash"] = summary["Total Value"] - sum(target_map.values())
        
        # Log generic "Rebalance"
        self._log_transaction("REBALANCE", "PORTFOLIO", 0, 0)
        
        self.save_state()
        return True

    def _log_transaction(self, action, asset, value, price):
        self.state["history"].append({
            "date": str(datetime.datetime.now()),
            "action": action,
            "asset": asset,
            "value": value,
            "price": price
        })
    
    def execute_predictive_rebalancing(self, predicted_weights: dict[str, float], current_prices: dict[str, float]):
        """
        Rebalance portfolio based on ML predictions.
        
        Args:
            predicted_weights: Dict of asset -> predicted optimal weight
            current_prices: Dict of asset -> current market price
        """
        summary = self.get_portfolio_summary()
        total_capital = summary["Total Value"]
        
        # Convert weights to dollar allocations
        target_allocations = {
            asset: weight * total_capital
            for asset, weight in predicted_weights.items()
        }
        
        # Execute rebalancing
        blueprint = pd.DataFrame([
            {"Item": asset, "Allocated Capital": allocation}
            for asset, allocation in target_allocations.items()
        ])
        
        self.execute_rebalancing(blueprint)
        LOGGER.info(f"Predictive rebalancing complete. {len(predicted_weights)} positions.")
    
    def get_performance_metrics(self) -> dict:
        """Calculate portfolio performance statistics."""
        history = pd.DataFrame(self.state["history"])
        
        if history.empty:
            return {
                "total_deposits": 0,
                "total_withdrawals": 0,
                "total_return_pct": 0
            }
        
        deposits = history[history["action"] == "DEPOSIT"]["value"].sum()
        summary = self.get_portfolio_summary()
        
        total_return = (summary["Total Value"] - deposits) / deposits if deposits > 0 else 0
        
        return {
            "total_deposits": deposits,
            "current_value": summary["Total Value"],
            "total_return_pct": total_return,
            "total_trades": len(history[history["action"] == "REBALANCE"])
        }

