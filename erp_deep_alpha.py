"""
ERP Deep Alpha Scanner
======================

A deeper scan that looks at item-level procurement costs and correlates
them with related commodity prices. The hypothesis is that your 
procurement costs for fertilizer/chemical inputs should correlate with
(or lead) the commodities you're exposed to.

Key Insight: Your existing correlation_report.txt shows strong 
correlations between items like NO3FE, CHEGLUCO, MRCCCN110 and
"weird data" like butter/cheese consumption. If these items' costs
correlate with external factors, those same factors may predict
commodity markets.

This scanner:
1. Fetches your top procurement items' monthly cost history
2. Fetches related commodity prices
3. Tests if YOUR costs predict MARKET prices

If your NO3FE costs correlate 0.96 with cheese consumption, and cheese
consumption correlates with corn prices, then YOUR costs may lead corn.

Usage:
    python erp_deep_alpha.py
"""

import datetime
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False

LOGGER = logging.getLogger(__name__)

# Commodities that may correlate with fertilizer/chemical procurement
COMMODITY_TARGETS = {
    # Agricultural (downstream demand for fertilizers)
    "corn": "ZC=F",
    "soybeans": "ZS=F",
    "wheat": "ZW=F",
    "soybean_meal": "ZM=F",
    
    # Energy (input costs)
    "natural_gas": "NG=F",
    "crude_oil": "CL=F",
    
    # Fertilizer proxies
    "cf_industries": "CF",  # Nitrogen producer
    "mosaic": "MOS",  # Phosphate/Potash
    "nutrien": "NTR",  # Diversified fertilizer
    "intrepid_potash": "IPI",  # Potash
    
    # Sector ETFs
    "materials": "XLB",
    "agriculture_etf": "MOO",
    "energy": "XLE",
}


@dataclass
class DeepCorrelation:
    """Result of correlation analysis."""
    item_number: str
    item_description: str
    commodity: str
    lag_months: int
    correlation: float
    p_value: float
    r_squared: float
    n_obs: int
    direction: str
    interpretation: str


class ERPDeepAlphaScanner:
    """
    Deep scanner that correlates item-level procurement costs with
    commodity market prices.
    """
    
    def __init__(self):
        self.cursor = None
        self._conn = None
        
        if HAS_PYODBC:
            try:
                from secrets_loader import build_connection_string
                conn_str, _, _, _ = build_connection_string()
                self._conn = pyodbc.connect(conn_str)
                self.cursor = self._conn.cursor()
                print("✅ Database connected")
            except Exception as e:
                print(f"⚠️  No database connection: {e}")
    
    def __del__(self):
        if self._conn:
            self._conn.close()
    
    def get_top_items_by_spend(self, top_n: int = 30) -> List[Tuple[str, str, float]]:
        """
        Get top items by total procurement spend.
        Returns: List of (item_number, description, total_spend)
        """
        if not self.cursor:
            return self._mock_top_items()
        
        query = """
        SELECT TOP (?)
            l.ITEMNMBR,
            i.ITEMDESC,
            SUM(l.EXTDCOST) AS TotalSpend
        FROM POP30310 l
        JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
        LEFT JOIN IV00101 i ON l.ITEMNMBR = i.ITEMNMBR
        WHERE h.RECEIPTDATE >= '2018-01-01'
          AND l.EXTDCOST > 0
        GROUP BY l.ITEMNMBR, i.ITEMDESC
        ORDER BY SUM(l.EXTDCOST) DESC
        """
        
        try:
            self.cursor.execute(query, (top_n,))
            rows = self.cursor.fetchall()
            return [
                (row.ITEMNMBR.strip(), 
                 (row.ITEMDESC or "").strip(), 
                 float(row.TotalSpend))
                for row in rows
            ]
        except Exception as e:
            LOGGER.error(f"Error getting top items: {e}")
            return self._mock_top_items()
    
    def get_item_monthly_costs(self, item_number: str) -> pd.Series:
        """
        Get monthly average unit cost for an item.
        Returns: pd.Series with DatetimeIndex (monthly)
        """
        if not self.cursor:
            return self._mock_monthly_costs(item_number)
        
        query = """
        SELECT 
            YEAR(h.RECEIPTDATE) AS Year,
            MONTH(h.RECEIPTDATE) AS Month,
            AVG(l.UNITCOST) AS AvgCost,
            SUM(l.EXTDCOST) AS TotalSpend,
            COUNT(*) AS NumReceipts
        FROM POP30300 h
        JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
        WHERE l.ITEMNMBR = ?
          AND h.RECEIPTDATE >= '2018-01-01'
          AND l.UNITCOST > 0
        GROUP BY YEAR(h.RECEIPTDATE), MONTH(h.RECEIPTDATE)
        HAVING COUNT(*) >= 1
        ORDER BY Year, Month
        """
        
        try:
            self.cursor.execute(query, (item_number,))
            rows = self.cursor.fetchall()
            
            if not rows:
                return pd.Series(dtype=float)
            
            data = {}
            for row in rows:
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                data[dt] = float(row.AvgCost)
            
            return pd.Series(data).sort_index()
            
        except Exception as e:
            LOGGER.error(f"Error getting costs for {item_number}: {e}")
            return pd.Series(dtype=float)
    
    def fetch_commodity_prices(self, start_date: str = "2018-01-01") -> pd.DataFrame:
        """
        Fetch monthly commodity prices.
        Returns: DataFrame with monthly prices for each commodity.
        """
        if not HAS_YFINANCE:
            return self._mock_commodity_prices()
        
        tickers = list(COMMODITY_TARGETS.values())
        print(f"Fetching {len(tickers)} commodity prices...")
        
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                end=datetime.date.today().isoformat(),
                progress=False
            )
            
            # Get adjusted close
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                prices = data
            
            # Resample to monthly (end of month)
            monthly = prices.resample('ME').last()
            
            # Rename columns to friendly names
            reverse_map = {v: k for k, v in COMMODITY_TARGETS.items()}
            monthly.columns = [reverse_map.get(str(c), str(c)) for c in monthly.columns]
            
            return monthly.dropna(how='all')
            
        except Exception as e:
            LOGGER.error(f"Error fetching commodities: {e}")
            return self._mock_commodity_prices()
    
    def calculate_correlation(
        self,
        item_costs: pd.Series,
        commodity_prices: pd.Series,
        lag: int = 0
    ) -> Optional[Dict]:
        """
        Calculate correlation between item costs and commodity prices.
        
        Args:
            item_costs: Monthly average unit costs
            commodity_prices: Monthly commodity prices
            lag: Months to lag costs (positive = costs lead prices)
        """
        if not HAS_SCIPY:
            return None
        
        if len(item_costs) < 12 or len(commodity_prices) < 12:
            return None
        
        # Shift costs if lagged (costs lead prices)
        if lag > 0:
            costs_aligned = item_costs.shift(lag)
        else:
            costs_aligned = item_costs
        
        # Find common dates
        common = costs_aligned.dropna().index.intersection(commodity_prices.dropna().index)
        
        if len(common) < 12:
            return None
        
        x = costs_aligned.loc[common].values
        y = commodity_prices.loc[common].values
        
        # Calculate metrics
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        # R-squared
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        return {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "r_squared": float(r_squared),
            "slope": float(slope),
            "n_obs": len(common),
        }
    
    def scan_all_correlations(
        self,
        min_r: float = 0.3,  # Lower threshold to find weaker signals
        max_p: float = 0.20,
        lags: List[int] = [0, 1, 2, 3],
        top_items: int = 30,
    ) -> List[DeepCorrelation]:
        """
        Scan all items against all commodities.
        """
        print("=" * 60)
        print("  ERP DEEP ALPHA SCANNER")
        print("=" * 60)
        
        # Get top items
        print(f"\n1. Getting top {top_items} items by spend...")
        items = self.get_top_items_by_spend(top_items)
        print(f"   Found {len(items)} items")
        
        # Get commodity prices
        print("\n2. Fetching commodity prices...")
        commodity_df = self.fetch_commodity_prices()
        print(f"   Got {len(commodity_df.columns)} commodities, {len(commodity_df)} months")
        
        # Scan correlations
        print(f"\n3. Scanning correlations (r>{min_r}, p<{max_p})...")
        results = []
        
        for item_num, item_desc, spend in items:
            # Get item costs
            costs = self.get_item_monthly_costs(item_num)
            
            if len(costs) < 12:
                continue
            
            for commodity_name in commodity_df.columns:
                prices = commodity_df[commodity_name].dropna()
                
                if len(prices) < 12:
                    continue
                
                for lag in lags:
                    corr = self.calculate_correlation(costs, prices, lag)
                    
                    if corr is None:
                        continue
                    
                    r = corr["pearson_r"]
                    p = corr["pearson_p"]
                    
                    if abs(r) >= min_r and p <= max_p:
                        direction = "positive" if r > 0 else "negative"
                        
                        # Interpretation
                        if lag > 0:
                            if r > 0:
                                interp = f"When {item_num} costs rise, {commodity_name} tends to rise {lag}mo later"
                            else:
                                interp = f"When {item_num} costs rise, {commodity_name} tends to fall {lag}mo later"
                        else:
                            if r > 0:
                                interp = f"{item_num} costs move WITH {commodity_name}"
                            else:
                                interp = f"{item_num} costs move OPPOSITE to {commodity_name}"
                        
                        results.append(DeepCorrelation(
                            item_number=item_num,
                            item_description=item_desc,
                            commodity=commodity_name,
                            lag_months=lag,
                            correlation=r,
                            p_value=p,
                            r_squared=corr["r_squared"],
                            n_obs=corr["n_obs"],
                            direction=direction,
                            interpretation=interp,
                        ))
        
        # Sort by absolute correlation
        results.sort(key=lambda x: abs(x.correlation), reverse=True)
        
        print(f"   Found {len(results)} significant correlations")
        
        return results
    
    def generate_report(self, results: List[DeepCorrelation]) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "ERP DEEP ALPHA DISCOVERY REPORT",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
        ]
        
        if not results:
            lines.append("No correlations found at current thresholds.")
            lines.append("Try lowering min_r or raising max_p.")
            return "\n".join(lines)
        
        lines.append(f"Found {len(results)} correlations\n")
        
        # Top 10 strongest
        lines.append("-" * 60)
        lines.append("TOP 10 STRONGEST CORRELATIONS")
        lines.append("-" * 60)
        
        for i, c in enumerate(results[:10], 1):
            lines.append(f"\n{i}. {c.item_number} → {c.commodity}")
            lines.append(f"   r={c.correlation:+.3f} | p={c.p_value:.4f} | R²={c.r_squared:.3f}")
            lines.append(f"   Lag: {c.lag_months}mo | N: {c.n_obs}")
            lines.append(f"   📊 {c.interpretation}")
        
        # Group by commodity (predictable)
        lines.append("\n" + "=" * 60)
        lines.append("CORRELATIONS BY COMMODITY")
        lines.append("=" * 60)
        
        by_commodity = {}
        for c in results:
            if c.commodity not in by_commodity:
                by_commodity[c.commodity] = []
            by_commodity[c.commodity].append(c)
        
        for commodity, corrs in sorted(by_commodity.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"\n--- {commodity.upper()} ({len(corrs)} signals) ---")
            for c in sorted(corrs, key=lambda x: abs(x.correlation), reverse=True)[:5]:
                lag_str = f"(Lag {c.lag_months}mo)" if c.lag_months > 0 else ""
                dir_sym = "↑" if c.direction == "positive" else "↓"
                lines.append(f"  {c.item_number:15} r={c.correlation:+.3f} {dir_sym} {lag_str}")
        
        # Lagged signals (PREDICTIVE!)
        lagged = [c for c in results if c.lag_months > 0]
        if lagged:
            lines.append("\n" + "=" * 60)
            lines.append("🔮 PREDICTIVE SIGNALS (Lagged Correlations)")
            lines.append("=" * 60)
            lines.append("These show ERP costs LEADING market prices!\n")
            
            for c in sorted(lagged, key=lambda x: abs(x.correlation), reverse=True)[:10]:
                lines.append(f"  {c.item_number} leads {c.commodity} by {c.lag_months}mo")
                lines.append(f"     r={c.correlation:+.3f} (p={c.p_value:.4f})")
                lines.append(f"     {c.interpretation}\n")
        
        lines.append("\n" + "=" * 60)
        lines.append("DISCLAIMER")
        lines.append("=" * 60)
        lines.append("Correlation does NOT imply causation or tradability.")
        lines.append("These findings require out-of-sample validation before use.")
        
        return "\n".join(lines)
    
    def run(self) -> Tuple[List[DeepCorrelation], str]:
        """Run full scan and return results + report."""
        results = self.scan_all_correlations()
        report = self.generate_report(results)
        
        # Save report
        with open("erp_deep_alpha_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 Report saved to erp_deep_alpha_report.txt")
        
        print("\n" + report)
        
        return results, report
    
    # Mock data for testing without DB
    def _mock_top_items(self) -> List[Tuple[str, str, float]]:
        """Mock top items for demo."""
        return [
            ("NO3FE", "Iron Nitrate", 500000),
            ("CHEGLUCO", "Glucose Chemical", 450000),
            ("NPKKNO3", "Potassium Nitrate", 400000),
            ("MRCCCN110", "Calcium Carbonate", 350000),
            ("NO3MN", "Manganese Nitrate", 300000),
            ("GRPSEAWEED", "Seaweed Extract", 280000),
            ("NO3MG60", "Magnesium Nitrate", 250000),
            ("NPKKOHDRY", "Potash Dry", 220000),
            ("NPKPHOS75", "Phosphoric 75%", 200000),
            ("NO3CA", "Calcium Nitrate", 180000),
        ]
    
    def _mock_monthly_costs(self, item_number: str) -> pd.Series:
        """Mock monthly costs with some trend."""
        np.random.seed(hash(item_number) % 1000)
        dates = pd.date_range("2018-01-01", periods=72, freq="MS")
        
        # Base price with trend and seasonality
        base = 0.50 + np.random.rand() * 0.5
        trend = np.linspace(0, 0.3, len(dates))
        seasonal = 0.1 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
        noise = np.random.randn(len(dates)) * 0.05
        
        values = base + trend + seasonal + noise
        return pd.Series(values, index=dates)
    
    def _mock_commodity_prices(self) -> pd.DataFrame:
        """Mock commodity prices for demo."""
        dates = pd.date_range("2018-01-01", periods=72, freq="MS")
        
        data = {}
        for name in ["corn", "soybeans", "natural_gas", "cf_industries", "materials"]:
            np.random.seed(hash(name) % 1000 + 42)
            base = 100 + np.random.rand() * 50
            trend = np.linspace(0, 30, len(dates))
            seasonal = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
            noise = np.random.randn(len(dates)) * 5
            data[name] = base + trend + seasonal + noise
        
        return pd.DataFrame(data, index=dates)


if __name__ == "__main__":
    scanner = ERPDeepAlphaScanner()
    results, report = scanner.run()
    
    print(f"\n✅ Scan complete! Found {len(results)} correlations.")
