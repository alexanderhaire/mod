
import logging
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from execution_robinhood import RobinhoodExecution
import warnings
warnings.filterwarnings('ignore')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alpha_hunt_omni.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)

class OmniStrategyLive:
    def __init__(self):
        self.tickers = [
            'SPY', 'TLT', 'GLD', 'XLE', 'UUP', # Trad
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', # Crypto 
            '^VIX' # Signal
        ]
        
    def fetch_data(self):
        LOGGER.info("Fetching real-time market data (1y)...")
        # Fetch enough data for 200MA (needs 200 trading days ~ 1 year)
        # Using 2y to be safe
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        try:
            data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
                except:
                    prices = data['Close']
            else:
                prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                
            return prices.ffill().dropna()
        except Exception as e:
            LOGGER.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def get_target_allocation(self, equity: float) -> dict:
        prices = self.fetch_data()
        if prices.empty:
            return {}
            
        current_prices = prices.iloc[-1]
        
        # --- 1. MACRO REGIME (The Golden Switch) ---
        spy = prices['SPY']
        ma200_spy = spy.rolling(200).mean().iloc[-1]
        curr_spy = spy.iloc[-1]
        
        is_bull_market = curr_spy > ma200_spy
        regime = "BULL (Attack)" if is_bull_market else "BEAR (Defend)"
        LOGGER.info(f"Market Regime: {regime} | SPY: {curr_spy:.2f} vs 200MA: {ma200_spy:.2f}")
        
        # --- 2. INFLATION REGIME (The ERP Switch) ---
        # Proxy: XLE Trend
        if 'XLE' in prices.columns:
            xle = prices['XLE']
            ma200_xle = xle.rolling(200).mean().iloc[-1]
            curr_xle = xle.iloc[-1]
            is_inflation = (curr_xle > ma200_xle)
            if is_inflation:
                LOGGER.info(f"Inflation Filter: ON (Commodities > Bonds). XLE: {curr_xle:.2f} > {ma200_xle:.2f}")
            else:
                LOGGER.info("Inflation Filter: OFF")
        else:
            is_inflation = False
            
        # --- 3. CRYPTO LOGIC ---
        btc = prices['BTC-USD']
        btc_vol = btc.pct_change().tail(30).std() * np.sqrt(365) * 100
        is_crypto_safe = btc_vol < 100
        
        LOGGER.info(f"BTC Volatility: {btc_vol:.1f}% | Safe: {is_crypto_safe}")
        
        crypto_assets = ['BTC-USD']
        if is_crypto_safe:
            # Check Altseason
            alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
            avail_alts = [c for c in alts if c in prices.columns]
            
            if avail_alts:
                btc_mom = btc.pct_change(14).iloc[-1]
                alt_mom = prices[avail_alts].pct_change(14).iloc[-1].mean()
                
                if alt_mom > btc_mom:
                    LOGGER.info(f"Altseason Detected! Alts ({alt_mom:.1%}) > BTC ({btc_mom:.1%})")
                    crypto_assets = avail_alts # Swap BTC for Alts
                else:
                    LOGGER.info(f"BTC Dominance. BTC ({btc_mom:.1%}) > Alts ({alt_mom:.1%})")
        else:
             LOGGER.warning("Crypto Volatility Too High. Moving Crypto allocation to Cash/Dollar.")
             crypto_assets = [] # No crypto
             
        # --- 4. ALLOCATION CALCULATION ---
        weights = {}
        
        # Crypto Sleeve Weight (Target 40% if Safe)
        w_crypto_total = 0.40 if is_crypto_safe else 0.0
        w_cash_total = 0.40 if not is_crypto_safe else 0.0 # Pivot to UUP if unsafe
        
        # Core Sleeve Weight (60%)
        # Normal weighting: 45 SPY / 10 TLT / 5 GLD
        # Bear weighting:   15 SPY / 35 TLT / 10 GLD
        
        if is_bull_market:
            weights['SPY'] = 0.45
            weights['TLT'] = 0.10
            weights['GLD'] = 0.05
        else:
            # Bear Market
            weights['SPY'] = 0.15
            weights['GLD'] = 0.10
            
            if is_inflation:
                # Inflation Bear: Swap TLT for XLE
                LOGGER.info("Defensive Pivot: Buying XLE (Energy) instead of TLT (Bonds)")
                weights['XLE'] = 0.35
            else:
                # Normal Bear: Buy TLT
                weights['TLT'] = 0.35
                
        # Fill Crypto Sleeve
        if w_crypto_total > 0 and crypto_assets:
            w_each = w_crypto_total / len(crypto_assets)
            for c in crypto_assets:
                weights[c] = w_each
        
        # Fill Cash Sleeve (UUP)
        if w_cash_total > 0:
            weights['UUP'] = w_cash_total
            
        # --- 5. FINALIZE ---
        # Convert to dollars
        targets = {}
        
        LOGGER.info("-" * 40)
        LOGGER.info("TARGET PORTFOLIO:")
        for asset, w in weights.items():
            amt = equity * w
            if amt > 1.0: # Min trade size
                targets[asset] = amt
                LOGGER.info(f"  {asset:<10} {w:>6.1%}  ${amt:,.2f}")
                
        return targets

def run():
    print("=" * 60)
    print("   ALPHA HUNT: OMNI STRATEGY LIVE")
    print("   Deploying the 'Golden Trinity' Architecture")
    print("=" * 60)
    
    # 1. Connect
    executor = RobinhoodExecution()
    if not executor.connected:
        print("\n❌ Login Failed. Check secrets.toml and TOTP.")
        return
        
    # 2. Status
    summary = executor.get_account_summary()
    equity = summary.get('equity', 0.0)
    print(f"\n💰 Equity: ${equity:,.2f}")
    
    if equity < 10: print("⚠️ Account empty."); return
    
    # 3. Strategy
    strategy = OmniStrategyLive()
    targets = strategy.get_target_allocation(equity)
    
    # 4. Execute
    if targets:
        print("\n⚖️ Rebalancing...")
        executor.rebalance_portfolio(targets)
        print("✅ Done.")
    else:
        print("⚠️ No allocation generated (Data error?).")

if __name__ == "__main__":
    run()
