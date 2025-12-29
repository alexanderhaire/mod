
import logging
import time
from typing import Dict, List, Any
import toml
from pathlib import Path
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
except ImportError:
    tradeapi = None

LOGGER = logging.getLogger(__name__)

class AlpacaExecution:
    """
    Handles execution of trades via Alpaca Markets API.
    Supports both Paper and Live trading modes.
    """
    def __init__(self, mode="paper"):
        self.mode = mode
        self.api = None
        self.connected = False
        self.headers = {}
        
        self.key_id = None
        self.secret_key = None
        self.base_url = None
        
        self._connect()
        
    def _connect(self):
        """Load keys and establish connection."""
        try:
            secrets_path = Path("secrets.toml")
            if not secrets_path.exists():
                LOGGER.warning("Alpaca: secrets.toml not found.")
                return

            config = toml.load(secrets_path)
            
            if "alpaca" not in config:
                LOGGER.warning("Alpaca: [alpaca] section missing in secrets.toml")
                return
                
            alpaca_conf = config["alpaca"]
            self.key_id = alpaca_conf.get("key", "")
            self.secret_key = alpaca_conf.get("secret", "")
            self.base_url = alpaca_conf.get("endpoint", "https://paper-api.alpaca.markets")
            
            if not self.key_id or not self.secret_key:
                LOGGER.warning("Alpaca: Missing Key ID or Secret.")
                return
                
            if tradeapi:
                self.api = tradeapi.REST(
                    self.key_id,
                    self.secret_key,
                    self.base_url,
                    api_version='v2'
                )
                
                # Check connection
                account = self.api.get_account()
                self.connected = True
                LOGGER.info(f"Connected to Alpaca ({account.status}): Equity ${float(account.equity):,.2f}")
            else:
                LOGGER.error("alpaca-trade-api not installed.")
                
        except Exception as e:
            LOGGER.error(f"Alpaca Connection Failed: {e}")
            self.connected = False

    def get_account_summary(self) -> Dict[str, Any]:
        """Return key account metrics."""
        if not self.connected:
            return {"status": "Disconnected", "equity": 0.0, "cash": 0.0}
            
        try:
            acct = self.api.get_account()
            return {
                "status": acct.status,
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "daytrade_count": acct.daytrade_count
            }
        except Exception as e:
            LOGGER.error(f"Alpaca Account Fetch Error: {e}")
            return {}

    def get_positions(self) -> Dict[str, float]:
        """Return dictionary of Symbol -> MarketValue."""
        if not self.connected:
            return {}
            
        try:
            positions = self.api.list_positions()
            pos_dict = {}
            for p in positions:
                # We track value for rebalancing
                pos_dict[p.symbol] = float(p.market_value)
            return pos_dict
        except Exception as e:
            LOGGER.error(f"Alpaca Positions Error: {e}")
            return {}

    def rebalance_portfolio(self, target_allocations: Dict[str, float]):
        """
        Rebalance the account to match the target dollar allocations.
        target_allocations: Dict of Symbol -> Target Market Value ($)
        """
        if not self.connected:
            LOGGER.warning("Alpaca: Cannot rebalance (Disconnected).")
            return
            
        LOGGER.info(f"⚖️ Starting Rebalance. Targets: {len(target_allocations)} positions.")
        
        # 1. Get Current State
        current_positions = self.get_positions() # Symbol -> $Value
        
        # 2. Calculate Deltas
        # Union of all symbols involved
        all_symbols = set(current_positions.keys()) | set(target_allocations.keys())
        
        orders_to_place = []
        
        for symbol in all_symbols:
            curr_val = current_positions.get(symbol, 0.0)
            target_val = target_allocations.get(symbol, 0.0)
            diff = target_val - curr_val
            
            # Sensitivity Threshold ($10 minimum to trade)
            if abs(diff) < 10.0:
                continue
                
            orders_to_place.append({
                "symbol": symbol,
                "diff": diff, # Positive = Buy, Negative = Sell
                "is_sell": diff < 0
            })
            
        # 3. Sort Orders: Sells FIRST (to free up buying power)
        orders_to_place.sort(key=lambda x: x["is_sell"], reverse=True)
        
        # 4. Execute
        for order in orders_to_place:
            symbol = order["symbol"]
            diff = order["diff"]
            side = "sell" if diff < 0 else "buy"
            qty_usd = abs(diff)
            
            LOGGER.info(f"Rebalance {symbol}: Current=${current_positions.get(symbol,0):.0f} -> Target=${target_allocations.get(symbol,0):.0f} | Action: {side.upper()} ${qty_usd:.2f}")
            
            # Submit
            self.execute_trade(symbol, qty_usd, side=side)
            
            # Tiny sleep to rate limit? Alpaca handles 200/min.
            time.sleep(0.1)
            
        LOGGER.info("⚖️ Rebalance Orders Submitted.")

    def execute_trade(self, symbol: str, notional_value: float, side: str = "buy"):
        """
        Submit a Market Order by Notional Value (fractional shares).
        """
        if not self.connected:
            return False
            
        try:
            # Safety: Min order size $1
            if notional_value < 1.0:
                return False
                
            LOGGER.info(f"Submitting Alpaca Order: {side.upper()} ${notional_value:.2f} of {symbol}")
            
            self.api.submit_order(
                symbol=symbol,
                notional=notional_value,
                side=side,
                type='market',
                time_in_force='day'
            )
            return True
        except Exception as e:
            LOGGER.error(f"Order Failed ({symbol}): {e}")
            return False
