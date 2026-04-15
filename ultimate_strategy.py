
"""
Ultimate Combined Strategy
==========================

Combining statistically validated findings:
1. Altseason Detector (p=0.018) - SIGNIFICANT ✅
2. VIX Term Structure (p=0.079) - Marginal
3. Crypto Composite (p=0.060) - Marginal

Strategy:
- Allocate across traditional assets (SPY/TLT/GLD) + crypto (BTC/alts)
- Use VIX term structure for equity timing
- Use Altseason detector for crypto timing

Risk Management (from PDF):
- Max crypto allocation: 40%
- Max single alt position: 12%
- If portfolio drawdown > 15%: reduce crypto to 20%
- If BTC volatility > 100% annualized: reduce crypto to 20%
- If VIX > 35: force defensive mode
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import alpaca_trade_api as tradeapi
import os
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (from PDF)
# =============================================================================

TRADITIONAL_ASSETS = ['SPY', 'QQQ', 'TLT', 'GLD', 'IEF']
CRYPTO_ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 'DOGE-USD']
VIX_SYMBOLS = ['^VIX', '^VIX3M']

# Risk limits from PDF
MAX_CRYPTO_ALLOCATION = 40.0  # Max 40% in crypto
MAX_SINGLE_ALT = 12.0  # Max 12% in any single altcoin
REDUCED_CRYPTO_ALLOCATION = 20.0  # Reduce to 20% when risk limits triggered
VIX_SPIKE_THRESHOLD = 35.0  # VIX > 35 = force defensive
BTC_VOL_THRESHOLD = 100.0  # BTC vol > 100% ann = reduce crypto


def fetch_yahoo_chart(symbol: str, days: int = 210) -> pd.DataFrame:
    """Fetch price data from Yahoo Finance Chart API (no library needed)"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {'interval': '1d', 'range': f'{days}d'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        result = data.get('chart', {}).get('result', [])
        if not result:
            return pd.DataFrame()
            
        quotes = result[0]
        timestamps = quotes.get('timestamp', [])
        ohlc = quotes.get('indicators', {}).get('quote', [{}])[0]
        
        if not timestamps or not ohlc.get('close'):
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'close': ohlc.get('close', []),
            'open': ohlc.get('open', []),
            'high': ohlc.get('high', []),
            'low': ohlc.get('low', []),
            'volume': ohlc.get('volume', []),
        })
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Yahoo fetch error for {symbol}: {e}")
        return pd.DataFrame()


# =============================================================================
# SIGNAL GENERATORS (matching PDF exactly)
# =============================================================================

def vix_term_structure_signal(vix_df: pd.DataFrame, vix3m_df: pd.DataFrame) -> pd.Series:
    """
    VIX term structure for equity timing.
    From PDF:
    - ratio = VIX / VIX3M
    - if ratio < 0.90: signal = CONTANGO (bullish, overweight SPY)
    - elif ratio > 1.05: signal = BACKWARDATION (bearish, overweight TLT)
    - else: signal = NEUTRAL
    """
    if vix_df.empty or vix3m_df.empty:
        return pd.Series(dtype=float)
    
    vix = vix_df['close']
    vix3m = vix3m_df['close']
    
    # Align indices
    common_idx = vix.index.intersection(vix3m.index)
    vix = vix.loc[common_idx]
    vix3m = vix3m.loc[common_idx]
    
    ratio = vix / vix3m
    ratio_smooth = ratio.rolling(5).mean()
    
    signal = pd.Series(0.0, index=common_idx)
    
    for i in range(min(60, len(signal)), len(signal)):
        if pd.notna(ratio_smooth.iloc[i]):
            if ratio_smooth.iloc[i] < 0.90:
                signal.iloc[i] = 1   # Contango = bullish (overweight SPY)
            elif ratio_smooth.iloc[i] > 1.05:
                signal.iloc[i] = -1  # Backwardation = bearish (overweight TLT)
    
    return signal


def altseason_signal(prices: pd.DataFrame, lookback: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Altseason detector for crypto timing.
    From PDF:
    - btc_momentum = BTC price.pct_change(14)
    - alt_momentum = mean([alt.pct_change(14) for alt in altcoins])
    - if alt_momentum > btc_momentum * 1.2: signal = ALTSEASON (buy top 3 alts)
    - else: signal = BTC_DOMINANCE (hold BTC)
    """
    alts = [c for c in prices.columns if c.endswith('-USD') and c != 'BTC-USD']
    
    if 'BTC-USD' not in prices.columns or len(alts) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=object)
    
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    
    signal = pd.Series(0.0, index=prices.index)
    top_alts_series = pd.Series([None] * len(prices), index=prices.index)
    
    for i in range(lookback + 10, len(prices)):
        # Calculate alt momentums
        alt_moms = {}
        for alt in alts:
            try:
                ret = prices[alt].iloc[i] / prices[alt].iloc[i-lookback] - 1
                if not np.isnan(ret):
                    alt_moms[alt] = ret
            except:
                pass
        
        avg_alt_mom = np.mean(list(alt_moms.values())) if len(alt_moms) > 0 else 0
        btc_m = btc_mom.iloc[i] if not np.isnan(btc_mom.iloc[i]) else 0
        
        if avg_alt_mom > btc_m * 1.2:
            signal.iloc[i] = 1  # Altseason
            # Get top 3 alts
            sorted_alts = sorted(alt_moms.items(), key=lambda x: x[1], reverse=True)[:3]
            top_alts_series.iloc[i] = [x[0] for x in sorted_alts]
        else:
            signal.iloc[i] = -1  # BTC dominance
    
    return signal, top_alts_series


def crypto_momentum_signal(prices: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """Overall crypto momentum signal based on BTC trend."""
    if 'BTC-USD' not in prices.columns:
        return pd.Series(dtype=float)
    
    btc = prices['BTC-USD']
    btc_ret = btc.pct_change(lookback)
    btc_ma = btc.rolling(lookback).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(lookback, len(prices)):
        if btc.iloc[i] > btc_ma.iloc[i] and btc_ret.iloc[i] > 0:
            signal.iloc[i] = 1
        elif btc.iloc[i] < btc_ma.iloc[i] and btc_ret.iloc[i] < 0:
            signal.iloc[i] = -1
    
    return signal


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class UltimateStrategy:
    """
    Ultimate Combined Strategy:
    - 60% traditional (SPY/TLT/GLD) - timed by VIX Term Structure
    - 40% crypto (BTC/Alts) - timed by Altseason + BTC Momentum
    
    Risk Limits (from PDF):
    - Max crypto: 40%
    - Max single alt: 12%
    - VIX > 35: force defensive
    - BTC vol > 100%: reduce crypto to 20%
    - Drawdown > 15%: reduce crypto to 20%
    """
    
    def __init__(self):
        base_url = os.getenv('ALPACA_BASE_URL') or os.getenv('ALPACA_ENDPOINT') or 'https://paper-api.alpaca.markets'
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_KEY'),
            secret_key=os.getenv('ALPACA_SECRET'),
            base_url=base_url,
            api_version='v2'
        )
        # Cache for signals
        self._prices = None
        self._vix = None
        self._vix3m = None
        self._last_fetch = None
        self._cache_duration = timedelta(minutes=5)
        self._portfolio_high_water_mark = None
    
    def _refresh_data(self):
        """Fetch fresh data using Alpaca Data API for real-time accuracy."""
        now = datetime.now()
        if self._last_fetch and (now - self._last_fetch) < self._cache_duration:
            return
        
        print("📊 Fetching real-time market data from Alpaca...")
        prices_dict = {}
        
        start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
        
        # 1. Traditional Assets via Alpaca
        try:
            # get_bars returns a dataframe with MultiIndex (symbol, timestamp) or just timestamp if single symbol
            bars = self.api.get_bars(TRADITIONAL_ASSETS, '1D', start=start_date).df
            if not bars.empty:
                for symbol in TRADITIONAL_ASSETS:
                    if 'symbol' in bars.columns:
                        df_sym = bars[bars['symbol'] == symbol]
                    else:
                        try:
                            df_sym = bars.xs(symbol)
                        except:
                            continue
                            
                    if not df_sym.empty:
                        s = df_sym['close']
                        if s.index.tz is not None:
                            s.index = s.index.tz_localize(None)
                        prices_dict[symbol] = s
                        
        except Exception as e:
            print(f"⚠️ Alpaca Stock Data Error: {e}")

        # 2. Crypto via Alpaca
        try:
            # Alpaca crypto symbols format for data: 'BTC/USD'
            alpaca_crypto = [s.replace('-USD', '/USD') for s in CRYPTO_ASSETS]
            
            cbars = self.api.get_crypto_bars(alpaca_crypto, '1D', start=start_date).df
            if not cbars.empty:
                 for sym in alpaca_crypto:
                    yahoo_sym = sym.replace('/USD', '-USD')
                    if 'symbol' in cbars.columns:
                        df_c = cbars[cbars['symbol'] == sym]
                    else:
                        try:
                            df_c = cbars.xs(sym)
                        except:
                            continue
                    
                    if not df_c.empty:
                        s = df_c['close']
                        if s.index.tz is not None:
                            s.index = s.index.tz_localize(None)
                        prices_dict[yahoo_sym] = s
                        
        except Exception as e:
            print(f"⚠️ Alpaca Crypto Data Error: {e}")
            
        # 3. VIX Indices (Use Yahoo as fallback/primary for indices)
        self._vix = fetch_yahoo_chart('^VIX', 300)
        self._vix3m = fetch_yahoo_chart('^VIX3M', 300)
        
        # 4. Fill gaps with Yahoo
        all_symbols = TRADITIONAL_ASSETS + CRYPTO_ASSETS
        for symbol in all_symbols:
            if symbol not in prices_dict:
                print(f"⚠️ Fetching fallback data for {symbol} from Yahoo...")
                df = fetch_yahoo_chart(symbol, 300)
                if not df.empty:
                    prices_dict[symbol] = df['close']
        
        if prices_dict:
            self._prices = pd.DataFrame(prices_dict)
            self._prices = self._prices.ffill().dropna()
            
        self._last_fetch = now
        print(f"   Data refreshed: {len(self._prices) if self._prices is not None else 0} rows")
    
    def _update_drawdown(self, current_equity: float) -> float:
        """Update high water mark and return current drawdown percentage."""
        if current_equity <= 0:
            return 0.0
            
        if self._portfolio_high_water_mark is None or current_equity > self._portfolio_high_water_mark:
            self._portfolio_high_water_mark = current_equity
            
        drawdown = (self._portfolio_high_water_mark - current_equity) / self._portfolio_high_water_mark
        return drawdown * 100

    def _get_btc_volatility(self) -> float:
        """Calculate BTC annualized volatility (for risk limit check)."""
        if self._prices is None or 'BTC-USD' not in self._prices.columns:
            return 50.0  # Default moderate vol
        
        btc = self._prices['BTC-USD']
        returns = btc.pct_change().dropna()
        if len(returns) < 20:
            return 50.0
        
        vol = returns.tail(30).std() * np.sqrt(365) * 100  # Crypto = 365 days
        return vol
    
    def _get_spot_vix(self) -> float:
        """Get current VIX level for risk check."""
        if self._vix is None or self._vix.empty:
            return 20.0
        return float(self._vix['close'].iloc[-1])
    
    def get_market_regime(self) -> Dict:
        """
        Risk Overlay using VIX term structure, SPY trend, and VIX spike check.
        PDF Rule: VIX > 35 = force defensive mode
        """
        self._refresh_data()
        
        try:
            # VIX term structure
            vix_signal = vix_term_structure_signal(self._vix, self._vix3m)
            current_vix_signal = vix_signal.iloc[-1] if len(vix_signal) > 0 else 0
            
            # Get spot VIX for spike check
            spot_vix = self._get_spot_vix()
            
            # SPY trend
            if self._prices is not None and 'SPY' in self._prices.columns:
                spy = self._prices['SPY']
                spy_ma200 = spy.rolling(min(200, len(spy))).mean().iloc[-1]
                spy_price = spy.iloc[-1]
                spy_above_ma = spy_price > spy_ma200
                
                # Calculate realized volatility
                returns = spy.pct_change().dropna()
                volatility = returns.tail(20).std() * np.sqrt(252) * 100
            else:
                spy_price = 0
                spy_ma200 = 0
                spy_above_ma = True
                volatility = 15
            
            # PDF RULE: VIX > 35 = force defensive mode
            vix_spike = spot_vix > VIX_SPIKE_THRESHOLD
            
            # Determine regime
            if vix_spike or current_vix_signal < 0 or volatility > 25 or not spy_above_ma:
                regime = 'RISK_OFF'
                reason = []
                if vix_spike:
                    reason.append(f'VIX SPIKE ({spot_vix:.1f} > {VIX_SPIKE_THRESHOLD})')
                if current_vix_signal < 0:
                    reason.append('VIX backwardation')
                if volatility > 25:
                    reason.append(f'High volatility ({volatility:.1f}%)')
                if not spy_above_ma:
                    reason.append('SPY below MA')
            else:
                regime = 'RISK_ON'
                reason = ['VIX contango' if current_vix_signal > 0 else 'Neutral VIX', 
                          f'Vol: {volatility:.1f}%']
            
            return {
                'status': regime,
                'vix': round(volatility, 2),
                'spot_vix': round(spot_vix, 2),
                'spy_price': round(spy_price, 2),
                'spy_ma200': round(spy_ma200, 2),
                'vix_signal': current_vix_signal,
                'vix_spike': vix_spike,
                'reason': ', '.join(reason)
            }
        except Exception as e:
            print(f"Regime error: {e}")
            return {'status': 'RISK_ON', 'reason': f'Error: {str(e)[:50]}', 
                    'vix': 0, 'spy_price': 0, 'spy_ma200': 0, 'vix_signal': 0,
                    'spot_vix': 0, 'vix_spike': False}
    
    def generate_signals(self) -> List[Dict]:
        """Generate signals for both traditional and crypto assets."""
        self._refresh_data()
        signals = []
        
        if self._prices is None:
            return signals
        
        # VIX signal for traditional assets
        vix_sig = vix_term_structure_signal(self._vix, self._vix3m)
        current_vix = vix_sig.iloc[-1] if len(vix_sig) > 0 else 0
        
        # Altseason and crypto momentum signals
        alt_sig, top_alts = altseason_signal(self._prices)
        crypto_mom = crypto_momentum_signal(self._prices)
        
        current_alt = alt_sig.iloc[-1] if len(alt_sig) > 0 else 0
        current_crypto_mom = crypto_mom.iloc[-1] if len(crypto_mom) > 0 else 0
        current_top_alts = top_alts.iloc[-1] if len(top_alts) > 0 and top_alts.iloc[-1] is not None else []
        
        # Traditional assets signals
        for symbol in ['SPY', 'TLT', 'GLD']:
            if symbol in self._prices.columns:
                score = 50  # Base score
                if symbol == 'SPY':
                    score += current_vix * 20  # VIX bullish = higher SPY
                elif symbol == 'TLT':
                    score -= current_vix * 15  # VIX bullish = lower bonds
                elif symbol == 'GLD':
                    score += 10  # Gold always a small allocation
                
                signals.append({
                    'symbol': symbol,
                    'score': round(max(0, min(99, score)), 1),
                    'type': 'traditional',
                    'signal': 'VIX' if symbol != 'GLD' else 'hedge'
                })
        
        # Crypto signals
        crypto_combined = (current_alt + current_crypto_mom) / 2
        
        if 'BTC-USD' in self._prices.columns:
            btc_score = 50 + crypto_combined * 25
            signals.append({
                'symbol': 'BTC-USD',
                'score': round(max(0, min(99, btc_score)), 1),
                'type': 'crypto',
                'signal': 'momentum'
            })
        
        # Add top alts if in altseason
        if current_alt > 0 and current_top_alts:
            for alt in current_top_alts[:3]:
                if alt in self._prices.columns:
                    signals.append({
                        'symbol': alt,
                        'score': round(60 + current_alt * 15, 1),
                        'type': 'crypto',
                        'signal': 'altseason'
                    })
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals
    
    def get_target_allocation(self, total_equity: float) -> Dict:
        """
        Main allocation logic (from PDF):
        
        Traditional Sleeve (60%):
        - Contango (bullish): 45% SPY, 10% TLT, 5% GLD
        - Neutral: 30% SPY, 22% TLT, 8% GLD
        - Backwardation (bearish): 15% SPY, 35% TLT, 10% GLD
        
        Crypto Sleeve (40%):
        - Strong Altseason: 10% BTC, 30% Top 3 Alts (10% each)
        - Moderate Altseason: 25% BTC, 15% ETH
        - Neutral: 20% BTC
        - Bearish: 5% BTC
        
        Risk Limits:
        - Max single alt: 12%
        - VIX > 35: force defensive
        - BTC vol > 100%: reduce crypto to 20%
        """
        self._refresh_data()
        regime = self.get_market_regime()
        
        # Get signals
        vix_sig = vix_term_structure_signal(self._vix, self._vix3m)
        vix_s = vix_sig.iloc[-1] if len(vix_sig) > 0 else 0
        
        alt_sig, top_alts = altseason_signal(self._prices) if self._prices is not None else (pd.Series(), pd.Series())
        crypto_mom = crypto_momentum_signal(self._prices) if self._prices is not None else pd.Series()
        
        alt_s = alt_sig.iloc[-1] if len(alt_sig) > 0 else 0
        crypto_s = crypto_mom.iloc[-1] if len(crypto_mom) > 0 else 0
        current_top_alts = top_alts.iloc[-1] if len(top_alts) > 0 and top_alts.iloc[-1] is not None else []
        
        allocation = {}
        crypto_reduced = False
        
        # === RISK OFF: Defensive allocation ===
        if regime['status'] == 'RISK_OFF':
            allocation = {'TLT': 50.0, 'GLD': 20.0, 'BIL': 30.0}
            return {
                'allocation': allocation,
                'leverage': 1.0,
                'regime': regime,
                'top_signals': [],
                'sleeve_traditional': 100.0,
                'sleeve_crypto': 0.0,
                'crypto_reduced': False
            }
        
        # === CHECK RISK LIMITS (Volatility & Drawdown) ===
        btc_vol = self._get_btc_volatility()
        drawdown = self._update_drawdown(total_equity)
        
        if btc_vol > BTC_VOL_THRESHOLD or drawdown > 15.0:
            crypto_reduced = True
            reason = []
            if btc_vol > BTC_VOL_THRESHOLD:
                reason.append(f"BTC volatility {btc_vol:.1f}%")
            if drawdown > 15.0:
                reason.append(f"Drawdown {drawdown:.1f}%")
            print(f"⚠️ [RISK] {' & '.join(reason)} > Threshold - reducing crypto to {REDUCED_CRYPTO_ALLOCATION}%")
        
        # === RISK ON: Active allocation ===
        
        # TRADITIONAL SLEEVE (60%) - from PDF table
        if vix_s > 0:
            # Contango (bullish): overweight SPY
            trad_w = {'SPY': 45.0, 'TLT': 10.0, 'GLD': 5.0}
        elif vix_s < 0:
            # Backwardation (bearish): overweight TLT
            trad_w = {'SPY': 15.0, 'TLT': 35.0, 'GLD': 10.0}
        else:
            # Neutral
            trad_w = {'SPY': 30.0, 'TLT': 22.0, 'GLD': 8.0}
        
        for asset, w in trad_w.items():
            allocation[asset] = w
        
        # CRYPTO SLEEVE (40% or 20% if reduced) - from PDF table
        max_crypto = REDUCED_CRYPTO_ALLOCATION if crypto_reduced else MAX_CRYPTO_ALLOCATION
        crypto_combined = (alt_s + crypto_s) / 2
        
        if crypto_combined > 0.5:
            # STRONG ALTSEASON: 10% BTC, 30% Alts (10% each, capped at 12%)
            allocation['BTC-USD'] = min(10.0, max_crypto * 0.25)
            if current_top_alts and len(current_top_alts) > 0:
                alt_weight = min(MAX_SINGLE_ALT, (max_crypto - 10.0) / min(3, len(current_top_alts)))
                for alt in current_top_alts[:3]:
                    allocation[alt] = round(alt_weight, 1)
        elif crypto_combined > 0:
            # MODERATE ALTSEASON: 25% BTC, 15% ETH
            btc_pct = min(25.0, max_crypto * 0.625)
            eth_pct = min(15.0, max_crypto * 0.375)
            allocation['BTC-USD'] = btc_pct
            allocation['ETH-USD'] = eth_pct
        elif crypto_combined > -0.5:
            # NEUTRAL: 20% BTC
            allocation['BTC-USD'] = min(20.0, max_crypto)
        else:
            # BEARISH: 5% BTC
            allocation['BTC-USD'] = min(5.0, max_crypto)
            # Add remaining to TLT
            remaining = max_crypto - 5.0
            if remaining > 0:
                allocation['TLT'] = allocation.get('TLT', 0) + remaining
        
        # Generate signals list
        signals = self.generate_signals()
        
        # Calculate actual sleeve totals
        trad_total = sum(allocation.get(a, 0) for a in ['SPY', 'TLT', 'GLD', 'IEF', 'BIL'])
        crypto_total = sum(allocation.get(a, 0) for a in allocation if a.endswith('-USD'))
        
        return {
            'allocation': allocation,
            'leverage': 1.0,  # No leverage in Ultimate Strategy
            'regime': regime,
            'top_signals': signals[:5],
            'sleeve_traditional': round(trad_total, 1),
            'sleeve_crypto': round(crypto_total, 1),
            'vix_signal': vix_s,
            'altseason_signal': alt_s,
            'crypto_momentum': crypto_s,
            'btc_volatility': round(btc_vol, 1),
            'crypto_reduced': crypto_reduced
        }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    # Load env vars from .env.local in parent dir
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local')
    load_dotenv(env_path)
    
    print("=" * 80)
    print("   ULTIMATE COMBINED STRATEGY - TEST")
    print("=" * 80)
    
    strategy = UltimateStrategy()
    
    print("\n📈 Market Regime:")
    regime = strategy.get_market_regime()
    print(f"   Status: {regime['status']}")
    print(f"   Reason: {regime['reason']}")
    print(f"   VIX Signal: {regime.get('vix_signal', 'N/A')}")
    print(f"   Spot VIX: {regime.get('spot_vix', 'N/A')}")
    print(f"   VIX Spike: {regime.get('vix_spike', False)}")
    
    print("\n📊 Signals:")
    signals = strategy.generate_signals()
    for sig in signals[:6]:
        print(f"   {sig['symbol']}: Score {sig['score']} ({sig['type']})")
    
    print("\n💰 Target Allocation:")
    alloc = strategy.get_target_allocation(100000)
    print(f"   Traditional: {alloc['sleeve_traditional']}%")
    print(f"   Crypto: {alloc['sleeve_crypto']}%")
    print(f"   BTC Volatility: {alloc.get('btc_volatility', 'N/A')}%")
    print(f"   Crypto Reduced: {alloc.get('crypto_reduced', False)}")
    for asset, weight in alloc['allocation'].items():
        print(f"   - {asset}: {weight}%")
    
    print("\n" + "=" * 80)
