
import datetime
import random
import logging
import pandas as pd
import yfinance as yf
from typing import Any, Dict, List
from constants import LOGGER

# Mapping UI Names to Yahoo Finance Tickers
TICKER_MAP = {
    # --- GLOBAL INDICES ---
    "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "Dow Jones": "^DJI", "Russell 2000": "^RUT", "Wilshire 5000": "^W5000", "VIX": "^VIX",
    "Nikkei 225 (Japan)": "^N225", "DAX (Germany)": "^GDAXI", "FTSE 100 (UK)": "^FTSE", "CAC 40 (France)": "^FCHI",
    "Euro Stoxx 50": "^STOXX50E", "STOXX 600 (Europe)": "^STOXX", "AEX (Netherlands)": "^AEX", "IBEX 35 (Spain)": "^IBEX", "FTSE MIB (Italy)": "FTSEMIB.MI",
    "Hang Seng (HK)": "^HSI", "Nifty 50 (India)": "^NSEI", "Shanghai Composite": "000001.SS", "CSI 300 (China)": "000300.SS", "Bovespa (Brazil)": "^BVSP",
    "KOSPI (Korea)": "^KS11", "ASX 200 (Australia)": "^AXJO", "TSX Composite (Canada)": "^GSPTSE", "JSX (Indonesia)": "^JKSE", "SET (Thailand)": "^SET.BK",
    "IPC (Mexico)": "^MXX", "MERVAL (Argentina)": "^MERV", "TASI (Saudi Arabia)": "^TASI.SR", "STI (Singapore)": "^STI",

    # --- ETFS ---
    "XLE (Energy)": "XLE", "XLB (Materials)": "XLB", "XLI (Industrials)": "XLI", "XLP (Consumer Staples)": "XLP", "XLY (Consumer Discretionary)": "XLY",
    "XLV (Health Care)": "XLV", "XLF (Financials)": "XLF", "XLK (Technology)": "XLK", "XLU (Utilities)": "XLU", "XLC (Communication Services)": "XLC", "IYR (Real Estate)": "IYR",
    "MOO (Agribusiness)": "MOO", "IYT (Transportation)": "IYT", "SMH (Semiconductors)": "SMH", "XBI (Biotech)": "XBI", "IBB (Biotech)": "IBB",
    "KRE (Regional Banks)": "KRE", "KBE (Banks)": "KBE", "ITB (Home Construction)": "ITB", "XHB (Homebuilders)": "XHB",
    "XME (Metals & Mining)": "XME", "JETS (Airlines)": "JETS", "TAN (Solar)": "TAN", "ICLN (Clean Energy)": "ICLN", "URA (Uranium)": "URA",
    "NLR (Nuclear)": "NLR", "LIT (Lithium)": "LIT", "REMX (Rare Earths)": "REMX", "COPX (Copper Miners)": "COPX", "SIL (Silver Miners)": "SIL",
    "GDX (Gold Miners)": "GDX", "GDXJ (Junior Gold Miners)": "GDXJ", "OIH (Oil Services)": "OIH", "XOP (Oil & Gas E&P)": "XOP",
    "AMLP (MLP Infrastructure)": "AMLP", "CORN (Corn Fund)": "CORN", "SOYB (Soybean Fund)": "SOYB", "WEAT (Wheat Fund)": "WEAT",
    "DBA (Agriculture Fund)": "DBA", "DBC (Commodity Index)": "DBC", "GLTR (Precious Metals)": "GLTR", "PALL (Palladium ETF)": "PALL",
    "WOOD (Timber)": "WOOD", "PHO (Water)": "PHO", "HACK (Cybersecurity)": "HACK", "SKYY (Cloud Computing)": "SKYY",
    "IGV (Software)": "IGV", "XRT (Retail)": "XRT", "IYZ (Telecom)": "IYZ", "PBW (Clean Energy)": "PBW", "KWEB (China Tech)": "KWEB",

    # --- LEVERAGED ETFs ---
    "TQQQ (ProShares UltraPro QQQ)": "TQQQ", "SQQQ (ProShares UltraPro Short QQQ)": "SQQQ",
    "SOXL (Direxion Daily Semi Bull 3X)": "SOXL", "SOXS (Direxion Daily Semi Bear 3X)": "SOXS",
    "UPRO (ProShares UltraPro S&P500)": "UPRO", "SPXU (ProShares UltraPro Short S&P500)": "SPXU",
    "TMF (Direxion Daily 20+ Yr Treasury Bull 3X)": "TMF", "TMV (Direxion Daily 20+ Yr Treasury Bear 3X)": "TMV",
    "LABU (Direxion Daily Biotech Bull 3X)": "LABU", "LABD (Direxion Daily Biotech Bear 3X)": "LABD",
    "YINN (Direxion Daily China Bull 3X)": "YINN", "YANG (Direxion Daily China Bear 3X)": "YANG",
    "BOIL (ProShares Ultra Bloomberg Natural Gas)": "BOIL", "KOLD (ProShares UltraShort Bloomberg Natural Gas)": "KOLD",

    # --- CURRENCIES ---
    "EUR/USD": "EURUSD=X", "JPY/USD": "JPY=X", "GBP/USD": "GBPUSD=X", "CAD/USD": "CAD=X", "AUD/USD": "AUDUSD=X", "NZD/USD": "NZDUSD=X", "CHF/USD": "CHF=X", "DXY (Dollar Index)": "DX-Y.NYB",
    "CNY/USD (Yuan)": "CNY=X", "MXN/USD (Peso)": "MXN=X", "BRL/USD (Real)": "BRL=X", "INR/USD (Rupee)": "INR=X",
    "RUB/USD (Ruble)": "RUB=X", "ZAR/USD (Rand)": "ZAR=X", "TRY/USD (Lira)": "TRY=X", "KRW/USD (Won)": "KRW=X", "SGD/USD (Sing Dollar)": "SGD=X",
    "HKD/USD (HK Dollar)": "HKD=X", "SEK/USD (Krona)": "SEK=X", "NOK/USD (Krone)": "NOK=X", "PLN/USD (Zloty)": "PLN=X", "HUF/USD (Forint)": "HUF=X",
    "CZK/USD (Koruna)": "CZK=X", "THB/USD (Baht)": "THB=X", "IDR/USD (Rupiah)": "IDR=X", "MYR/USD (Ringgit)": "MYR=X", "PHP/USD (Peso)": "PHP=X",
    "VND/USD (Dong)": "VND=X", "CLP/USD (Chilean Peso)": "CLP=X", "COP/USD (Col Peso)": "COP=X", "PEN/USD (Sol)": "PEN=X", "ARS/USD (Argentina)": "ARS=X",

    # --- METALS ---
    "Gold (Comex)": "GC=F", "Silver (Comex)": "SI=F", "Platinum": "PL=F", "Palladium": "PA=F", "Copper (Comex)": "HG=F", 
    
    # --- ENERGY ---
    "Crude Oil (WTI)": "CL=F", "Brent Crude (ICE)": "BZ=F", "Natural Gas (Henry Hub)": "NG=F", "Natural Gas (TTF Dutch)": "TTF=F", "JKM (LNG Asia)": "JKM=F",
    "Heating Oil (ULSD)": "HO=F", "RBOB Gasoline": "RB=F", "Ethanol (CBOT)": "CU=F", 
    
    # --- AGRICULTURE ---
    "Corn (CBOT)": "ZC=F", "Soybeans (CBOT)": "ZS=F", "Soybean Meal (CBOT)": "ZM=F", "Soybean Oil (CBOT)": "ZL=F",
    "Wheat (SRW CBOT)": "ZW=F", "Wheat (Matif Milling)": "KE=F", # Proxy KC Wheat
    "Cocoa (ICE NY)": "CC=F", "Coffee (Arabica)": "KC=F", 
    "Sugar #11 (Raw)": "SB=F", "Cotton #2": "CT=F", "Orange Juice (FCOJ)": "OJ=F", 
    
    # --- CRYPTO ---
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Tether (USDT)": "USDT-USD", "BNB (BNB)": "BNB-USD", "Solana (SOL)": "SOL-USD", "USDC (USDC)": "USDC-USD", "XRP (XRP)": "XRP-USD", "Dogecoin (DOGE)": "DOGE-USD",
    "Cardano (ADA)": "ADA-USD", "Shiba Inu (SHIB)": "SHIB-USD", "Avalanche (AVAX)": "AVAX-USD", "TRON (TRX)": "TRX-USD", "Polkadot (DOT)": "DOT-USD", "Bitcoin Cash (BCH)": "BCH-USD", "Chainlink (LINK)": "LINK-USD",
    "Polygon (MATIC)": "MATIC-USD", "Litecoin (LTC)": "LTC-USD", "Internet Computer (ICP)": "ICP-USD", "Uniswap (UNI)": "UNI-USD", "Ethereum Classic (ETC)": "ETC-USD", "Filecoin (FIL)": "FIL-USD",
    "Bitcoin Futures (CME)": "BTC=F", "Ethereum Futures (CME)": "ETH=F", "BITO (ETF)": "BITO", "GBTC": "GBTC", "ETHE": "ETHE",
    
    # --- STOCKS ---
    "Dow Inc (DOW)": "DOW", "LyondellBasell (LYB)": "LYB", "Westlake (WLK)": "WLK", "Eastman (EMN)": "EMN", "Celanese (CE)": "CE",
    "BASF (Germany)": "BAS.DE", "Bayer (Germany)": "BAYN.DE", "Nutrien (NTR)": "NTR", "Mosaic (MOS)": "MOS", "CF Industries (CF)": "CF",
    "Yara International (Norway)": "YAR.OL", "Corteva (CTVA)": "CTVA", "FMC Corp": "FMC", "Albemarle (ALB)": "ALB",
    "ExxonMobil (XOM)": "XOM", "Chevron (CVX)": "CVX", "Shell (SHEL)": "SHEL", "BP": "BP", "TotalEnergies": "TTE",
    "Caterpillar (CAT)": "CAT", "Union Pacific (UNP)": "UNP", "FedEx": "FDX", "UPS": "UPS"
}

def get_market_context() -> list:
    """Fallback if needed."""
    return []

def get_usage_forecasts(item_number: str) -> Dict[str, Any]:
    """Stub."""
    return {}

def fetch_agricultural_market_data(commodity: str, timeframe: str = "6m") -> Dict[str, Any]:
    """
    Fetch REAL market data from Yahoo Finance.
    """
    ticker = TICKER_MAP.get(commodity)
    
    # 1. Fallback for unmapped assets: Use a proxy or return empty
    if not ticker:
        # Try to guess if it's a known stock symbol in parens e.g. "Apple (AAPL)" -> AAPL
        import re
        match = re.search(r'\((.*?)\)', commodity)
        if match:
            ticker = match.group(1)
        else:
            # If still nothing, return a dummy structure to prevent app crash, 
            # but log warning.
            return {
                "commodity": commodity,
                "current_price_index": 100.0,
                "trend": "neutral",
                "volatility": "low",
                "forecast": "flat",
                "data": []
            }
            
    # 2. Fetch Data
    try:
        # Map timeframe to yfinance period
        period = "6mo"
        if timeframe == "1y": period = "1y"
        elif timeframe == "5y": period = "5y"
        elif timeframe == "1mo": period = "1mo"
        
        df = yf.download(ticker, period=period, progress=False, timeout=10)
        
        if df.empty:
            raise ValueError("No data returned")
            
        # 3. Format Response
        # Handle MultiIndex columns (e.g. ('Adj Close', 'AAPL')) or Single Index ('Adj Close')
        # Flatten columns if multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Select best price column
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if price_col not in df.columns:
             # Fallback: take the first column if typical names missing
             price_col = df.columns[0]
             
        current_price = float(df[price_col].iloc[-1])
        start_price = float(df[price_col].iloc[0])
        
        # Calculate simple trend
        change = (current_price - start_price) / start_price
        trend = "increasing" if change > 0.05 else "decreasing" if change < -0.05 else "stable"
        
        # Calculate Volatility
        rets = df[price_col].pct_change().dropna()
        vol = rets.std() * (252 ** 0.5) # Annualized
        vol_str = "high" if vol > 0.3 else "medium" if vol > 0.15 else "low"
        
        # Format Data Points
        trends = []
        for date, row in df.iterrows():
            # yfinance returns pandas Timestamp in index
            try:
                price_val = float(row[price_col])
            except:
                continue # Skip bad rows
                
            trends.append({
                'date': date.strftime('%Y-%m-%d'),
                'month': date.strftime('%B %Y'),
                'commodity': commodity,
                'price_index': round(price_val, 2),
                'volume_index': 100 # Placeholder
            })
            
        return {
            "commodity": commodity,
            "current_price_index": round(current_price, 2),
            "trend": trend,
            "volatility": vol_str,
            "forecast": "Unknown", # No forecast for real data
            "data": trends
        }
        
    except Exception as e:
        LOGGER.warning(f"Failed to fetch real data for {commodity} ({ticker}): {e}")
        return {
            "commodity": commodity,
            "current_price_index": 0.0,
            "trend": "error",
            "volatility": "error",
            "forecast": "error",
            "data": []
        }

def fetch_market_data_pool(commodities: List[str], _progress_callback=None) -> Dict[str, Dict[str, Any]]:
    """
    Batch fetch data for multiple commodities using yfinance batch download.
    This is significantly faster and more robust than looping.
    """
    # 1. Gather Tickers
    ticker_map_reverse = {}
    valid_tickers = []
    
    for comm in commodities:
        t = TICKER_MAP.get(comm)
        # Strick matching only: do NOT guess tickers to avoid 404s on "Countries" or "Indices"
        if t:
            valid_tickers.append(t)
            ticker_map_reverse[comm] = t
            
    if not valid_tickers:
        return {}

    # 2. Batch Download (2y history)
    try:
        # group_by='ticker' ensures we get a MultiIndex with Ticker at level 0
        batch_df = yf.download(valid_tickers, period="2y", group_by='ticker', progress=False, timeout=20, threads=True)
    except Exception as e:
        LOGGER.error(f"Batch download failed: {e}")
        return {}

    pool = {}
    total_items = len(commodities)
    
    # 3. Process Each Commodity
    for i, comm in enumerate(commodities):
        ticker = ticker_map_reverse.get(comm)
        res = {
                "commodity": comm,
                "current_price_index": 0.0,
                "trend": "error",
                "volatility": "error",
                "forecast": "error",
                "data": []
            }

        if ticker and not batch_df.empty:
            try:
                # Extract data for this ticker
                # If only 1 ticker requested, yfinance returns flat DF (no MultiIndex)
                if len(valid_tickers) == 1:
                    df = batch_df
                else:
                    if ticker in batch_df.columns.get_level_values(0):
                        df = batch_df[ticker]
                    else:
                        df = pd.DataFrame() # Ticker missing from batch result

                if not df.empty:
                    # Logic matches fetch_agricultural_market_data
                    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                    if price_col not in df.columns and len(df.columns) > 0:
                        price_col = df.columns[0]
                    
                    if price_col in df.columns:
                        # Clean failures
                        df = df.dropna(subset=[price_col])
                        
                        if not df.empty:
                            current_price = float(df[price_col].iloc[-1])
                            start_price = float(df[price_col].iloc[0])
                            
                            # Trend
                            change = (current_price - start_price) / start_price if start_price != 0 else 0
                            trend = "increasing" if change > 0.05 else "decreasing" if change < -0.05 else "stable"
                            
                            # Volatility
                            rets = df[price_col].pct_change().dropna()
                            vol = rets.std() * (252 ** 0.5)
                            vol_str = "high" if vol > 0.3 else "medium" if vol > 0.15 else "low"
                            
                            # Data Points
                            trends = []
                            for date, row in df.iterrows():
                                try:
                                    price_val = float(row[price_col])
                                    trends.append({
                                        'date': date.strftime('%Y-%m-%d'),
                                        'month': date.strftime('%B %Y'),
                                        'commodity': comm,
                                        'price_index': round(price_val, 2),
                                        'volume_index': 100
                                    })
                                except:
                                    continue
                                    
                            res = {
                                "commodity": comm,
                                "current_price_index": round(current_price, 2),
                                "trend": trend,
                                "volatility": vol_str,
                                "forecast": "Unknown",
                                "data": trends
                            }

            except Exception as inner_e:
                LOGGER.warning(f"Error processing batch data for {comm}: {inner_e}")
        
        pool[comm] = res
        
        if _progress_callback:
            _progress_callback((i + 1) / total_items)

    return pool
