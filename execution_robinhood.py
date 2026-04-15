
import robin_stocks.robinhood as r
import pyotp
import logging
import time
from typing import Dict, List, Any
import toml
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class RobinhoodExecution:
    """
    Handles execution of trades via Robinhood API (unofficial).
    Supports Stocks and Crypto with fractional shares.
    Requires 'robin_stocks' and 'pyotp'.
    """
    def __init__(self):
        self.connected = False
        self.username = None
        self.password = None
        self.totp_secret = None
        
        self._connect()
        
    def _connect(self):
        """Login to Robinhood using TOTP for MFA."""
        try:
            secrets_path = Path("secrets.toml")
            if not secrets_path.exists():
                LOGGER.warning("Robinhood: secrets.toml not found.")
                return

            config = toml.load(secrets_path)
            
            if "robinhood" not in config:
                LOGGER.warning("Robinhood: [robinhood] section missing in secrets.toml")
                return
                
            rh_conf = config["robinhood"]
            self.username = rh_conf.get("username", "")
            self.password = rh_conf.get("password", "")
            self.totp_secret = rh_conf.get("totp_secret", "").replace(" ", "")
            
            if not self.username or not self.password or not self.totp_secret:
                LOGGER.warning("Robinhood: Missing credentials.")
                return
            
            # Generate MFA code
            totp = pyotp.TOTP(self.totp_secret)
            mfa_code = totp.now()
            
            # Login
            login = r.login(self.username, self.password, mfa_code=mfa_code)
            
            if login:
                self.connected = True
                profile = r.load_account_profile()
                LOGGER.info(f"Connected to Robinhood: {profile.get('market_value', 'Unknown')}")
            else:
                LOGGER.error("Robinhood Login Failed.")
                
        except Exception as e:
            LOGGER.error(f"Robinhood Connection Failed: {e}")
            self.connected = False

    def get_account_summary(self) -> Dict[str, Any]:
        """Return key account metrics."""
        if not self.connected:
            return {"status": "Disconnected", "equity": 0.0, "cash": 0.0}
            
        try:
            profile = r.load_portfolio_profile()
            crypto = r.load_crypto_profile()
            
            equity = float(profile.get('equity', 0.0))
            cash = float(profile.get('withdrawable_amount', 0.0))
            buying_power = float(profile.get('buying_power', 0.0))
            
            return {
                "status": "Active",
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power
            }
        except Exception as e:
            LOGGER.error(f"Robinhood Account Fetch Error: {e}")
            return {}

    def get_positions(self) -> Dict[str, float]:
        """
        Return dictionary of Symbol -> MarketValue ($).
        Combines Equities and Crypto.
        """
        if not self.connected:
            return {}
            
        pos_dict = {}
        try:
            # 1. Equities
            my_stocks = r.build_holdings()
            for symbol, data in my_stocks.items():
                val = float(data.get('equity', 0.0))
                if val > 1.0: # Ignore dust
                    pos_dict[symbol] = val
            
            # 2. Crypto
            my_crypto = r.get_crypto_positions()
            for c in my_crypto:
                qty = float(c.get('quantity_available', 0.0))
                if qty > 0:
                    currency = c.get('currency', {}).get('code', '')
                    # Get price (cached/fast)
                    price_info = r.get_crypto_quote(currency)
                    price = float(price_info.get('mark_price', 0.0))
                    val = qty * price
                    if val > 1.0:
                        # Normalize symbol format to match strategy: 'BTC-USD'
                        sym = f"{currency}-USD" 
                        pos_dict[sym] = val
                        
            return pos_dict
            
        except Exception as e:
            LOGGER.error(f"Robinhood Positions Error: {e}")
            return {}

    def rebalance_portfolio(self, target_allocations: Dict[str, float]):
        """
        Rebalance the account.
        target_allocations: Dict of Symbol -> Target Market Value ($)
        """
        if not self.connected:
            LOGGER.warning("Robinhood: Cannot rebalance (Disconnected).")
            return
            
        LOGGER.info(f"⚖️ RH Starting Rebalance. Targets: {len(target_allocations)} positions.")
        
        # 1. Get Current State
        current_positions = self.get_positions()
        
        # 2. Calculate Deltas
        all_symbols = set(current_positions.keys()) | set(target_allocations.keys())
        orders = []
        
        for symbol in all_symbols:
            curr_val = current_positions.get(symbol, 0.0)
            target_val = target_allocations.get(symbol, 0.0)
            diff = target_val - curr_val
            
            if abs(diff) < 5.0: # $5 threshold for RH
                continue
                
            orders.append({
                "symbol": symbol,
                "diff": diff,
                "is_sell": diff < 0
            })
            
        # 3. Sort Orders: Sells FIRST
        orders.sort(key=lambda x: x["is_sell"], reverse=True)
        
        # 4. Execute
        for order in orders:
            symbol = order["symbol"]
            diff = order["diff"]
            qty_usd = abs(diff)
            
            self.execute_trade(symbol, qty_usd, side="sell" if diff < 0 else "buy")
            time.sleep(1.0) # Be gentle with unofficial API

    def execute_trade(self, symbol: str, notional_value: float, side: str = "buy") -> bool:
        """Submit fractional order by dollar amount."""
        if not self.connected: return False
        
        if notional_value < 1.0: return False
        
        LOGGER.info(f"RH Order: {side.upper()} ${notional_value:.2f} of {symbol}")
        
        try:
            # Check if Crypto
            if "-USD" in symbol or symbol in ['BTC','ETH','DOGE','SOL','ADA','AVAX']:
                # Clean symbol for RH Crypto (e.g. 'BTC-USD' -> 'BTC')
                crypto_sym = symbol.replace('-USD', '')
                if side == "buy":
                    r.order_buy_crypto_by_price(crypto_sym, notional_value)
                else:
                    r.order_sell_crypto_by_price(crypto_sym, notional_value)
            else:
                # Stock
                if side == "buy":
                    r.order_buy_fractional_by_price(symbol, notional_value)
                else:
                    r.order_sell_fractional_by_price(symbol, notional_value)
            
            return True
        except Exception as e:
            LOGGER.error(f"RH Trade Failed ({symbol}): {e}")
            return False
