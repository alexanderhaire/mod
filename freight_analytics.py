import json
import os
import pandas as pd
import numpy as np
import re
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import streamlit as st

# Constants
BROKER_QUOTES_FILE = "broker_quotes.jsonl"
PLANT_CITY_ZIP = "33563"
PLANT_CITY_COORDS = (28.0195, -82.1252) # Plant City, FL

# Known freight rate benchmarks ($/ton by origin zip or region)
# Some routes are flat rate (per load), others are per-ton
KNOWN_RATES = {
    # Mulberry area (short haul ~20-30 miles) - PER TON
    "33860": 11.50,  # Mulberry, FL
    "33801": 11.50,  # Lakeland, FL
    "33803": 11.50,  # Lakeland, FL
    
    # Cape Canaveral area (longer haul ~100 miles) - PER TON
    "32920": 28.00,  # Cape Canaveral, FL
    "32931": 28.00,  # Cocoa Beach, FL
    "32922": 28.00,  # Cocoa, FL
    "32926": 28.00,  # Cocoa, FL
    
    # Atlanta area (~450 miles) - typically FLAT RATE ~$3000/load
    "30374": 75.00,  # Atlanta (SQM) - est $3000/40 tons
    "30301": 75.00,  # Atlanta
    "30303": 75.00,  # Atlanta
    "30318": 75.00,  # Atlanta (Bottle Crew)
    
    # Michigan (Bottle Crew) - FLAT RATE via PLS Logistics
    "48325": 100.00,  # West Bloomfield, MI (est $3000/30 tons)
    "48322": 100.00,  # West Bloomfield, MI
    
    # Default regional estimates
    "DEFAULT_LOCAL": 12.00,   # <50 miles
    "DEFAULT_REGIONAL": 20.00, # 50-150 miles
    "DEFAULT_LONG": 30.00,    # >150 miles
}

# Flat rate routes (not per-ton) - VERIFIED from ERP
FLAT_RATE_ROUTES = {
    # Atlanta -> Plant City (Bottle Crew via Pittsburgh Logistics)
    # VERIFIED: PITTSBURGHLOGIS payments = $1,650 (Dec 2025)
    "30374": 1650.00,  # Atlanta (verified)
    "30318": 1650.00,  # Atlanta
    "30301": 1650.00,  # Atlanta general
    
    # Michigan -> Plant City (Bottle Crew via Pittsburgh Logistics)
    "48325": 1650.00,  # West Bloomfield, MI (Bottle Crew)
    "48322": 1650.00,  # West Bloomfield, MI
}

def get_rate_per_ton(origin_zip: str) -> float:
    """Get the known or estimated per-ton rate for a given origin."""
    if origin_zip in KNOWN_RATES:
        return KNOWN_RATES[origin_zip]
    # Could add distance-based estimation here
    return KNOWN_RATES.get("DEFAULT_REGIONAL", 20.00)

def estimate_tonnage(freight_price: float, origin_zip: str) -> float:
    """Estimate tonnage based on freight price and known per-ton rate."""
    rate = get_rate_per_ton(origin_zip)
    if rate > 0 and freight_price > 0:
        return round(freight_price / rate, 1)
    return 0.0

def calculate_per_ton_rate(freight_price: float, tonnage: float) -> float:
    """Calculate the per-ton rate from total price and tonnage."""
    if tonnage > 0:
        return round(freight_price / tonnage, 2)
    return 0.0

# Freight carrier to material vendor mapping with CORRECT ship-from origins
# Format: Carrier -> [(Vendor, ship-from zip, ship-from city)]
FREIGHT_LINK_CONFIG = {
    # Pittsburgh Logistics hauls from Atlanta, GA (NOT Michigan HQ)
    "PITTSBURGHLOGIS": [("BOTTLECREW", "30301", "Atlanta, GA")],
    
    # David Cole hauls SQM from Cape Canaveral/Atlanta area
    "DAVIDCOLE": [("SQM", "32920", "Cape Canaveral, FL")],
    
    # Jesse Cole also hauls SQM
    "JESSE COLE TRUC": [("SQM", "32920", "Cape Canaveral, FL")],
}

def load_linked_freight_data(cursor=None) -> pd.DataFrame:
    """
    Load freight data by linking freight carrier invoices to material vendor POs.
    
    This function:
    1. Gets freight carrier invoices (e.g., Pittsburgh Logistics)
    2. Gets material vendor PO receipt dates (e.g., Bottle Crew)
    3. Links them by date proximity (+/- 14 days)
    4. Returns actual freight costs matched to material origins
    """
    if cursor is None:
        try:
            from db_pool import get_connection
            with get_connection() as conn:
                return load_linked_freight_data(cursor=conn.cursor())
        except Exception as e:
            print(f"Could not connect to ERP: {e}")
            return pd.DataFrame()
    
    linked_records = []
    
    for freight_carrier, vendor_configs in FREIGHT_LINK_CONFIG.items():
        # 1. Get freight invoices for this carrier
        cursor.execute(f"""
            SELECT DOCDATE, DOCNUMBR, DOCAMNT, VENDORID
            FROM PM30200 
            WHERE VENDORID = '{freight_carrier}'
            ORDER BY DOCDATE DESC
        """)
        freight_invoices = cursor.fetchall()
        
        if not freight_invoices:
            continue
        
        # vendor_configs is now list of (vendor_id, origin_zip, origin_city) tuples
        for vendor_config in vendor_configs:
            vendor_id, origin_zip, origin_city = vendor_config
            
            # 2. Get material PO dates (just need dates for matching)
            cursor.execute(f"""
                SELECT DISTINCT CAST(h.RECEIPTDATE AS DATE) as po_date
                FROM POP30300 h
                WHERE h.VENDORID = '{vendor_id}'
                ORDER BY po_date DESC
            """)
            po_dates = [row.po_date for row in cursor.fetchall()]
            
            # 3. Match invoices to PO dates (within 14 days)
            for inv in freight_invoices:
                inv_date = inv.DOCDATE.date() if hasattr(inv.DOCDATE, 'date') else inv.DOCDATE
                
                for po_date in po_dates:
                    days_diff = abs((inv_date - po_date).days)
                    if days_diff <= 14:  # Within 2 weeks
                        linked_records.append({
                            "submitted_at": inv.DOCDATE,
                            "broker_id": freight_carrier,
                            "origin_zip": origin_zip,  # USE CONFIG ORIGIN, not vendor HQ
                            "origin_city": origin_city,
                            "freight_price": float(inv.DOCAMNT),
                            "material_vendor": vendor_id,
                            "days_offset": days_diff,
                            "source": "ERP_Linked",
                            "doc_number": inv.DOCNUMBR
                        })
                        break  # Only match once per invoice
    
    if not linked_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(linked_records)
    df["dest_zip"] = PLANT_CITY_ZIP
    df["day_of_week"] = pd.to_datetime(df["submitted_at"]).dt.day_name()
    
    return df

def load_erp_freight_history(cursor=None) -> pd.DataFrame:
    """
    Load historical freight data from ERP Purchase Receipts.
    Pulls ORFRTAMT (freight amount) from POP30300 with vendor zip codes.
    """
    if cursor is None:
        try:
            from db_pool import get_connection
            with get_connection() as conn:
                return load_erp_freight_history(cursor=conn.cursor())
        except Exception as e:
            print(f"Could not connect to ERP: {e}")
            return pd.DataFrame()
    
    try:
        # Query historical freight from Purchase Receipts
        query = """
        SELECT 
            h.POPRCTNM as receipt_num,
            h.RECEIPTDATE as submitted_at,
            h.VENDORID as broker_id,
            h.VENDNAME as broker_name,
            v.ZIPCODE as origin_zip,
            h.ORFRTAMT as freight_price,
            v.CITY as origin_city,
            v.STATE as origin_state
        FROM POP30300 h
        LEFT JOIN PM00200 v ON h.VENDORID = v.VENDORID
        WHERE h.ORFRTAMT > 0
          AND h.RECEIPTDATE >= DATEADD(year, -3, GETDATE())
        ORDER BY h.RECEIPTDATE DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Clean up
        df["origin_zip"] = df["origin_zip"].apply(lambda x: str(x).strip()[:5] if x else None)
        df["broker_id"] = df["broker_id"].apply(lambda x: str(x).strip() if x else "Unknown")
        df["freight_price"] = pd.to_numeric(df["freight_price"], errors='coerce').fillna(0.0)
        df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors='coerce')
        df["dest_zip"] = PLANT_CITY_ZIP
        df["source"] = "ERP"  # Mark data source
        
        # Add time features
        df["day_of_week"] = df["submitted_at"].dt.day_name()
        df["hour_of_day"] = 12  # Default for receipts (no time captured)
        df["date_only"] = df["submitted_at"].dt.date
        
        # Filter valid
        df = df[df["freight_price"] > 0]
        df = df.dropna(subset=["origin_zip"])
        
        return df
        
    except Exception as e:
        print(f"Error loading ERP freight: {e}")
        return pd.DataFrame()

def load_freight_data() -> pd.DataFrame:
    """
    Load broker quotes from JSONL file into a DataFrame.
    Returns empty DataFrame if file doesn't exist.
    """
    if not os.path.exists(BROKER_QUOTES_FILE):
        return pd.DataFrame()
    
    quotes = []
    try:
        with open(BROKER_QUOTES_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        quotes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading quotes: {e}")
        return pd.DataFrame()

    if not quotes:
        return pd.DataFrame()

    df = pd.DataFrame(quotes)
    
    # Normalize Columns
    if "broker_id" not in df.columns: df["broker_id"] = "Unknown"
    if "freight_price" not in df.columns: df["freight_price"] = 0.0
    if "vendor_quote_summary" not in df.columns: df["vendor_quote_summary"] = {}

    # Extract nested fields
    # We need Origin Zip. Usually in 'location' or 'origin' inside summary
    def _extract_origin(summary):
        if not isinstance(summary, dict): return None
        # Try explicit origin field first
        loc = summary.get("origin") or summary.get("location")
        if not loc: return None
        
        # Extract Zip
        match = re.search(r'\b\d{5}\b', str(loc))
        if match:
            return match.group(0)
        return None

    df["origin_zip"] = df["vendor_quote_summary"].apply(_extract_origin)
    df["dest_zip"] = PLANT_CITY_ZIP # Currently all inbound to PC (Assumption)

    # Convert price
    df["freight_price"] = pd.to_numeric(df["freight_price"], errors='coerce').fillna(0.0)

    # Filter out invalid rows (no price or no origin)
    df_clean = df.dropna(subset=["origin_zip", "freight_price"])
    df_clean = df_clean[df_clean["freight_price"] > 0]

    # Parse Dates
    if "submitted_at" in df_clean.columns:
        df_clean["submitted_at"] = pd.to_datetime(df_clean["submitted_at"], errors='coerce')
        df_clean["day_of_week"] = df_clean["submitted_at"].dt.day_name()
        df_clean["hour_of_day"] = df_clean["submitted_at"].dt.hour
        df_clean["date_only"] = df_clean["submitted_at"].dt.date
    else:
        # Fallback if column missing
        df_clean["submitted_at"] = pd.NaT
        df_clean["day_of_week"] = "Unknown"
        df_clean["hour_of_day"] = 0
        df_clean["date_only"] = None
    
    # Mark source
    df_clean["source"] = "Quote"

    return df_clean


def load_all_freight_data() -> pd.DataFrame:
    """
    Load and combine freight data from RELIABLE sources only:
    1. Manual broker quotes (broker_quotes.jsonl)
    2. Linked freight (carrier invoices matched to material POs by date)
    
    NOTE: ORFRTAMT from POP30300 is DISABLED - it shows per-unit freight
    allocations ($6-$30), NOT actual freight costs ($800-$1,650).
    Real freight comes from carrier invoices (PM30200) linked by date.
    
    Returns a single DataFrame with all historical freight records.
    """
    # 1. Load manual broker quotes
    df_quotes = load_freight_data()
    
    # 2. Load LINKED freight (carrier invoices -> material POs)
    # This is the REAL freight data from carriers like Pittsburgh Logistics, David Cole
    df_linked = load_linked_freight_data()
    
    # NOTE: ORFRTAMT from POP30300 is NOT used - it's not actual freight costs
    # df_erp = load_erp_freight_history()  # DISABLED - incorrect data
    
    # Combine reliable sources only
    if df_quotes.empty and df_linked.empty:
        return pd.DataFrame()
    
    # Ensure consistent columns before concat
    common_cols = ["origin_zip", "dest_zip", "broker_id", "freight_price", 
                   "submitted_at", "day_of_week", "source"]
    
    frames = []
    if not df_quotes.empty:
        for col in common_cols:
            if col not in df_quotes.columns:
                df_quotes[col] = None
        frames.append(df_quotes[common_cols])
    
    if not df_linked.empty:
        for col in common_cols:
            if col not in df_linked.columns:
                df_linked[col] = None
        frames.append(df_linked[common_cols])
    
    df_combined = pd.concat(frames, ignore_index=True)
    
    # Sort by date
    df_combined = df_combined.sort_values("submitted_at", ascending=False)
    
    return df_combined

def analyze_routes(df: pd.DataFrame):
    """
    Aggregate data by Route (Origin -> Dest) to find stats.
    Returns a DataFrame with route stats.
    """
    if df.empty:
        return pd.DataFrame()

    # Group by Origin Zip
    # We assume Dest is always Plant City for now
    
    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "Unknown"
    
    stats = df.groupby(["origin_zip", "broker_id"]).agg(
        avg_price=("freight_price", "mean"),
        min_price=("freight_price", "min"),
        max_price=("freight_price", "max"),
        quote_count=("freight_price", "count"),
        primary_source=("source", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
    ).reset_index()

    # Find Best Broker per Route
    # Sort by Origin, then Min Price
    stats = stats.sort_values(by=["origin_zip", "min_price"])
    
    # Create a summary per route
    route_summary = []
    for origin, group in stats.groupby("origin_zip"):
        best_row = group.iloc[0] # Cheapest min_price
        
        # Determine primary source for this route
        # Get sources from the original data for this origin
        origin_data = df[df["origin_zip"] == origin]
        erp_count = len(origin_data[origin_data["source"] == "ERP"])
        quote_count = len(origin_data[origin_data["source"] == "Quote"])
        
        if erp_count > 0 and quote_count == 0:
            source = "ERP"  # Verified (actually paid)
        elif quote_count > 0 and erp_count == 0:
            source = "Quote"  # Manual entry
        else:
            source = "Mixed"  # Both sources
        
        route_summary.append({
            "origin_zip": origin,
            "dest_zip": PLANT_CITY_ZIP,
            "best_broker": best_row["broker_id"],
            "best_price": best_row["min_price"],
            "avg_market_price": group["avg_price"].mean(), 
            "competitors": group["broker_id"].nunique(),
            "sample_size": group["quote_count"].sum(),
            "source": source  # Track data source
        })
        
    return pd.DataFrame(route_summary)

def predict_best_broker(origin_zip: str, df_history: pd.DataFrame = None):
    """
    Given an origin zip, return the best historical broker.
    """
    if df_history is None:
        df_history = load_freight_data()
        
    if df_history.empty:
        return None

    # Filter for exact match
    matches = df_history[df_history["origin_zip"] == origin_zip]
    
    if matches.empty:
        # TODO: Implement "Near Neighbor" logic here if needed (Geopy distance)
        # For now, return None
        return None

    # Rank brokers
    broker_stats = matches.groupby("broker_id")["freight_price"].agg(["mean", "min", "count"]).reset_index()
    # Score: Prefer lower price. Tie-breaker: frequency.
    broker_stats = broker_stats.sort_values(by=["mean", "count"], ascending=[True, False])
    
    best = broker_stats.iloc[0]
    return {
        "broker_id": best["broker_id"],
        "predicted_price": best["mean"],
        "confidence_samples": int(best["count"]),
        "all_candidates": broker_stats.to_dict("records")
    }

# --- Geocoding Cache ---
# Simple in-memory cache for the session
COORD_CACHE = {
    PLANT_CITY_ZIP: PLANT_CITY_COORDS
}

def get_coordinates(zip_code: str):
    """
    Get (lat, lon) for a zip code. Uses simple caching.
    """
    if zip_code in COORD_CACHE:
        return COORD_CACHE[zip_code]
        
    try:
        geolocator = Nominatim(user_agent="cdi_freight_predictor_v1")
        # Limit US to avoid global ambiguity
        location = geolocator.geocode({"postalcode": zip_code, "country": "US"}, timeout=10)
        
        if location:
            coords = (location.latitude, location.longitude)
            COORD_CACHE[zip_code] = coords
            return coords
    except Exception as e:
        print(f"Geocoding error for {zip_code}: {e}")
        
    return None
