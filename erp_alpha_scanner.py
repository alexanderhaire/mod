"""
ERP Alpha Scanner
==================

Mines operational ERP data (Great Plains/Dynamics GP) for alpha signals
that may correlate with market returns.

Signals Extracted:
1. Procurement Volume/Spend (monthly aggregates)
2. Procurement Price Changes (YoY unit cost changes)
3. Sales Velocity (monthly revenue trends)
4. Inventory Turnover Changes
5. Purchase Lead Time Trends
6. Order Backlog Levels

These signals are tested for correlation with:
- Agricultural/Chemical commodities
- Sector ETFs (XLB, MOO, XLE)
- Related stocks (LYB, DOW, MOS, NTR, CF)

Usage:
    scanner = ERPAlphaScanner(cursor)
    report = scanner.run_full_scan()
    print(report)
"""

import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyodbc

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

LOGGER = logging.getLogger(__name__)

# Market tickers to correlate ERP data against
MARKET_TARGETS = {
    # Agricultural/Fertilizer Commodities
    "corn": "ZC=F",
    "soybeans": "ZS=F",
    "wheat": "ZW=F",
    "natural_gas": "NG=F",
    "urea_cf": "CF",  # CF Industries (urea proxy)
    "potash_mos": "MOS",  # Mosaic (potash proxy)
    "nutrien": "NTR",
    
    # Sector ETFs
    "materials_xlb": "XLB",
    "agriculture_moo": "MOO",
    "energy_xle": "XLE",
    
    # Chemical Stocks
    "lyondell": "LYB",
    "dow": "DOW",
    "eastman": "EMN",
}


@dataclass
class ERPSignal:
    """A time-series signal extracted from ERP data."""
    name: str
    description: str
    frequency: str  # 'monthly', 'weekly'
    data: pd.Series  # DatetimeIndex -> float value
    source_tables: List[str]


@dataclass
class AlphaCorrelation:
    """Result of correlating an ERP signal with a market ticker."""
    erp_signal: str
    market_ticker: str
    lag_months: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_observations: int
    direction: str  # "positive" or "negative"
    
    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant at 5% level."""
        return self.pearson_p < 0.05 and abs(self.pearson_r) > 0.5


class ERPAlphaScanner:
    """
    Scans ERP database for potential alpha signals.
    
    Methodology:
    1. Extract monthly time-series from GP tables
    2. Download corresponding market data
    3. Calculate correlations with various lags
    4. Report statistically significant findings
    """
    
    def __init__(self, cursor: pyodbc.Cursor = None):
        """
        Initialize scanner.
        
        Args:
            cursor: Optional pyodbc cursor. If None, will try to create connection.
        """
        self.cursor = cursor
        self._owns_connection = False
        
        if cursor is None:
            try:
                from secrets_loader import build_connection_string
                conn_str, _, _, _ = build_connection_string()
                self._conn = pyodbc.connect(conn_str)
                self.cursor = self._conn.cursor()
                self._owns_connection = True
            except Exception as e:
                LOGGER.warning(f"Could not create DB connection: {e}")
                
    def __del__(self):
        if self._owns_connection and hasattr(self, '_conn'):
            self._conn.close()
    
    # =========================================================================
    # SIGNAL EXTRACTION
    # =========================================================================
    
    def extract_procurement_spend(self, start_year: int = 2018) -> ERPSignal:
        """
        Extract monthly total procurement spend from purchase receipts.
        
        Signal: Total $ received per month (proxy for input demand)
        """
        if not self.cursor:
            return self._mock_signal("procurement_spend")
            
        query = """
        SELECT 
            YEAR(h.RECEIPTDATE) AS Year,
            MONTH(h.RECEIPTDATE) AS Month,
            SUM(l.EXTDCOST) AS TotalSpend
        FROM POP30300 h
        JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
        WHERE h.RECEIPTDATE >= ?
          AND l.EXTDCOST > 0
        GROUP BY YEAR(h.RECEIPTDATE), MONTH(h.RECEIPTDATE)
        ORDER BY Year, Month
        """
        
        try:
            start_date = datetime.date(start_year, 1, 1)
            self.cursor.execute(query, (start_date,))
            rows = self.cursor.fetchall()
            
            if not rows:
                LOGGER.warning("No procurement data found")
                return self._mock_signal("procurement_spend")
            
            # Convert to monthly time series
            data = {}
            for row in rows:
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                data[dt] = float(row.TotalSpend)
            
            series = pd.Series(data).sort_index()
            
            return ERPSignal(
                name="procurement_spend",
                description="Total monthly procurement spend ($)",
                frequency="monthly",
                data=series,
                source_tables=["POP30300", "POP30310"]
            )
            
        except Exception as e:
            LOGGER.error(f"Error extracting procurement spend: {e}")
            return self._mock_signal("procurement_spend")
    
    def extract_procurement_volume_by_category(self, start_year: int = 2018) -> Dict[str, ERPSignal]:
        """
        Extract monthly procurement volume by item class.
        
        Returns dict of signals, one per major item class.
        """
        if not self.cursor:
            return {"raw_materials": self._mock_signal("raw_materials_volume")}
            
        query = """
        SELECT 
            YEAR(h.RECEIPTDATE) AS Year,
            MONTH(h.RECEIPTDATE) AS Month,
            i.ITMCLSCD AS ItemClass,
            SUM(l.EXTDCOST) AS TotalSpend,
            SUM(CASE WHEN l.UNITCOST > 0 THEN l.EXTDCOST / l.UNITCOST ELSE 0 END) AS TotalQty
        FROM POP30300 h
        JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
        LEFT JOIN IV00101 i ON l.ITEMNMBR = i.ITEMNMBR
        WHERE h.RECEIPTDATE >= ?
          AND l.EXTDCOST > 0
        GROUP BY YEAR(h.RECEIPTDATE), MONTH(h.RECEIPTDATE), i.ITMCLSCD
        ORDER BY Year, Month, i.ITMCLSCD
        """
        
        try:
            start_date = datetime.date(start_year, 1, 1)
            self.cursor.execute(query, (start_date,))
            rows = self.cursor.fetchall()
            
            # Group by item class
            class_data: Dict[str, Dict[pd.Timestamp, float]] = {}
            for row in rows:
                cls = (row.ItemClass or "UNKNOWN").strip()
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                
                if cls not in class_data:
                    class_data[cls] = {}
                class_data[cls][dt] = float(row.TotalSpend)
            
            # Create signals for top classes by total volume
            signals = {}
            for cls, data in class_data.items():
                series = pd.Series(data).sort_index()
                if len(series) >= 12:  # Need at least 1 year of data
                    safe_name = cls.lower().replace(" ", "_").replace("-", "_")
                    signals[safe_name] = ERPSignal(
                        name=f"procurement_{safe_name}",
                        description=f"Monthly procurement spend for {cls}",
                        frequency="monthly",
                        data=series,
                        source_tables=["POP30300", "POP30310", "IV00101"]
                    )
            
            return signals
            
        except Exception as e:
            LOGGER.error(f"Error extracting category procurement: {e}")
            return {}
    
    def extract_sales_velocity(self, start_year: int = 2018) -> ERPSignal:
        """
        Extract monthly sales volume from invoices.
        
        Signal: Total invoice $ per month (proxy for demand/activity)
        """
        if not self.cursor:
            return self._mock_signal("sales_velocity")
            
        query = """
        SELECT 
            YEAR(h.DOCDATE) AS Year,
            MONTH(h.DOCDATE) AS Month,
            SUM(CASE 
                WHEN h.SOPTYPE = 4 THEN -ABS(l.XTNDPRCE)  -- Returns 
                ELSE ABS(l.XTNDPRCE) 
            END) AS NetSales
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        WHERE h.DOCDATE >= ?
          AND h.SOPTYPE IN (3, 4)  -- Invoice, Return
        GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE)
        ORDER BY Year, Month
        """
        
        try:
            start_date = datetime.date(start_year, 1, 1)
            self.cursor.execute(query, (start_date,))
            rows = self.cursor.fetchall()
            
            data = {}
            for row in rows:
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                data[dt] = float(row.NetSales) if row.NetSales else 0
            
            series = pd.Series(data).sort_index()
            
            return ERPSignal(
                name="sales_velocity",
                description="Net monthly sales revenue ($)",
                frequency="monthly",
                data=series,
                source_tables=["SOP30200", "SOP30300"]
            )
            
        except Exception as e:
            LOGGER.error(f"Error extracting sales velocity: {e}")
            return self._mock_signal("sales_velocity")
    
    def extract_inventory_turnover(self, start_year: int = 2018) -> ERPSignal:
        """
        Extract monthly inventory turnover signal.
        
        Signal: (Outflows / Avg Inventory) per month - higher = faster turns
        """
        if not self.cursor:
            return self._mock_signal("inventory_turnover")
            
        # This is a simplified proxy - actual turnover calculation would need
        # point-in-time inventory snapshots
        query = """
        SELECT 
            YEAR(h.DOCDATE) AS Year,
            MONTH(h.DOCDATE) AS Month,
            SUM(CASE WHEN t.TRXQTY < 0 THEN ABS(t.TRXQTY * t.UNITCOST) ELSE 0 END) AS COGS,
            SUM(CASE WHEN t.TRXQTY > 0 THEN ABS(t.TRXQTY * t.UNITCOST) ELSE 0 END) AS Inflows
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        WHERE h.DOCDATE >= ?
        GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE)
        ORDER BY Year, Month
        """
        
        try:
            start_date = datetime.date(start_year, 1, 1)
            self.cursor.execute(query, (start_date,))
            rows = self.cursor.fetchall()
            
            data = {}
            for row in rows:
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                # Use COGS as proxy for inventory velocity
                data[dt] = float(row.COGS) if row.COGS else 0
            
            series = pd.Series(data).sort_index()
            
            return ERPSignal(
                name="inventory_turns_proxy",
                description="Monthly COGS ($ value of inventory consumed)",
                frequency="monthly",
                data=series,
                source_tables=["IV30300", "IV30200"]
            )
            
        except Exception as e:
            LOGGER.error(f"Error extracting inventory turnover: {e}")
            return self._mock_signal("inventory_turnover")
    
    def extract_purchase_lead_time(self, start_year: int = 2018) -> ERPSignal:
        """
        Extract average purchase lead time (PO date to receipt date).
        
        Signal: Avg days to receive goods - rising may signal supply tightness
        """
        if not self.cursor:
            return self._mock_signal("purchase_lead_time")
            
        query = """
        SELECT 
            YEAR(r.RECEIPTDATE) AS Year,
            MONTH(r.RECEIPTDATE) AS Month,
            AVG(DATEDIFF(day, po.DOCDATE, r.RECEIPTDATE)) AS AvgLeadTime,
            COUNT(*) AS NumReceipts
        FROM POP30310 l
        JOIN POP30300 r ON l.POPRCTNM = r.POPRCTNM
        JOIN POP30100 po ON l.PONUMBER = po.PONUMBER
        WHERE r.RECEIPTDATE >= ?
          AND l.PONUMBER <> ''
          AND r.RECEIPTDATE >= po.DOCDATE
          AND DATEDIFF(day, po.DOCDATE, r.RECEIPTDATE) BETWEEN 0 AND 365
        GROUP BY YEAR(r.RECEIPTDATE), MONTH(r.RECEIPTDATE)
        HAVING COUNT(*) >= 5
        ORDER BY Year, Month
        """
        
        try:
            start_date = datetime.date(start_year, 1, 1)
            self.cursor.execute(query, (start_date,))
            rows = self.cursor.fetchall()
            
            data = {}
            for row in rows:
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                data[dt] = float(row.AvgLeadTime) if row.AvgLeadTime else 0
            
            series = pd.Series(data).sort_index()
            
            return ERPSignal(
                name="purchase_lead_time",
                description="Average days from PO to receipt",
                frequency="monthly",
                data=series,
                source_tables=["POP30100", "POP30300", "POP30310"]
            )
            
        except Exception as e:
            LOGGER.error(f"Error extracting lead time: {e}")
            return self._mock_signal("purchase_lead_time")
    
    def extract_order_backlog(self, start_year: int = 2018) -> ERPSignal:
        """
        Extract monthly average open order backlog.
        
        Signal: Sum of unshipped order value - rising = strong demand
        
        Note: This reconstructs historical backlog from shipped orders,
        as GP doesn't maintain point-in-time snapshots easily.
        """
        if not self.cursor:
            return self._mock_signal("order_backlog")
            
        # Approximate by looking at time between order and ship
        query = """
        SELECT 
            YEAR(h.DOCDATE) AS Year,
            MONTH(h.DOCDATE) AS Month,
            COUNT(DISTINCT h.SOPNUMBE) AS OrderCount,
            SUM(l.XTNDPRCE) AS OrderValue,
            AVG(DATEDIFF(day, h.DOCDATE, h.GLPOSTDT)) AS AvgFulfillDays
        FROM SOP30200 h
        JOIN SOP30300 l ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE
        WHERE h.SOPTYPE = 3  -- Invoice (shipped orders)
          AND h.DOCDATE >= ?
        GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE)
        ORDER BY Year, Month
        """
        
        try:
            start_date = datetime.date(start_year, 1, 1)
            self.cursor.execute(query, (start_date,))
            rows = self.cursor.fetchall()
            
            data = {}
            for row in rows:
                dt = pd.Timestamp(year=row.Year, month=row.Month, day=1)
                # Use order value as proxy for demand level
                data[dt] = float(row.OrderValue) if row.OrderValue else 0
            
            series = pd.Series(data).sort_index()
            
            return ERPSignal(
                name="order_value",
                description="Monthly order value ($)",
                frequency="monthly",
                data=series,
                source_tables=["SOP30200", "SOP30300"]
            )
            
        except Exception as e:
            LOGGER.error(f"Error extracting order backlog: {e}")
            return self._mock_signal("order_backlog")
    
    def extract_all_signals(self, start_year: int = 2018) -> List[ERPSignal]:
        """Extract all available ERP signals."""
        signals = []
        
        # Core signals
        signals.append(self.extract_procurement_spend(start_year))
        signals.append(self.extract_sales_velocity(start_year))
        signals.append(self.extract_inventory_turnover(start_year))
        signals.append(self.extract_purchase_lead_time(start_year))
        signals.append(self.extract_order_backlog(start_year))
        
        # Category-level signals
        category_signals = self.extract_procurement_volume_by_category(start_year)
        signals.extend(category_signals.values())
        
        # Filter out empty signals
        signals = [s for s in signals if s is not None and len(s.data) >= 12]
        
        LOGGER.info(f"Extracted {len(signals)} ERP signals")
        return signals
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def fetch_market_data(
        self, 
        start_date: str = "2018-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch market data for correlation targets.
        
        Returns DataFrame with monthly returns for each ticker.
        """
        if not HAS_YFINANCE:
            LOGGER.warning("yfinance not available, using mock data")
            return self._mock_market_data()
        
        end_date = end_date or datetime.date.today().isoformat()
        
        tickers = list(MARKET_TARGETS.values())
        print(f"Fetching market data for {len(tickers)} tickers...")
        
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            # Get adjusted close prices
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                prices = data
            
            # Resample to monthly (end of month)
            monthly_prices = prices.resample('ME').last()
            
            # Calculate monthly returns
            monthly_returns = monthly_prices.pct_change().dropna()
            
            # Create friendly column names
            reverse_map = {v: k for k, v in MARKET_TARGETS.items()}
            monthly_returns.columns = [reverse_map.get(c, c) for c in monthly_returns.columns]
            
            return monthly_returns
            
        except Exception as e:
            LOGGER.error(f"Error fetching market data: {e}")
            return self._mock_market_data()
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    
    @staticmethod
    def calculate_correlation(
        signal_data: pd.Series,
        market_data: pd.Series,
        lag: int = 0
    ) -> Optional[AlphaCorrelation]:
        """
        Calculate correlation between ERP signal and market returns.
        
        Args:
            signal_data: Monthly ERP signal values
            market_data: Monthly market returns
            lag: Number of months to lag the signal (positive = signal leads market)
        
        Returns:
            AlphaCorrelation result or None if insufficient data
        """
        if not HAS_SCIPY:
            LOGGER.warning("scipy not available for correlation")
            return None
        
        # Align data
        if lag > 0:
            # Signal leads market: shift signal back
            signal_aligned = signal_data.shift(lag)
        else:
            signal_aligned = signal_data
        
        # Find common dates
        common_idx = signal_aligned.dropna().index.intersection(market_data.dropna().index)
        
        if len(common_idx) < 12:
            return None
        
        x = signal_aligned.loc[common_idx].values
        y = market_data.loc[common_idx].values
        
        # Calculate correlations
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        direction = "positive" if pearson_r > 0 else "negative"
        
        return AlphaCorrelation(
            erp_signal=signal_data.name if hasattr(signal_data, 'name') else "unknown",
            market_ticker=market_data.name if hasattr(market_data, 'name') else "unknown",
            lag_months=lag,
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_r=float(spearman_r),
            spearman_p=float(spearman_p),
            n_observations=len(common_idx),
            direction=direction
        )
    
    def scan_correlations(
        self,
        signals: List[ERPSignal],
        market_data: pd.DataFrame,
        lags: List[int] = [0, 1, 2, 3],
        min_r: float = 0.5,
        max_p: float = 0.10
    ) -> List[AlphaCorrelation]:
        """
        Scan all signals against all market tickers for correlations.
        
        Args:
            signals: List of ERP signals
            market_data: DataFrame of monthly market returns
            lags: Lags to test (in months)
            min_r: Minimum absolute correlation to include
            max_p: Maximum p-value to include
        
        Returns:
            List of significant correlations
        """
        results = []
        
        for signal in signals:
            signal_series = signal.data.copy()
            signal_series.name = signal.name
            
            # Normalize signal to changes (like returns)
            signal_changes = signal_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            
            for market_col in market_data.columns:
                market_series = market_data[market_col].copy()
                market_series.name = market_col
                
                for lag in lags:
                    corr = self.calculate_correlation(signal_changes, market_series, lag)
                    
                    if corr is None:
                        continue
                    
                    if abs(corr.pearson_r) >= min_r and corr.pearson_p <= max_p:
                        results.append(corr)
        
        # Sort by absolute correlation strength
        results.sort(key=lambda x: abs(x.pearson_r), reverse=True)
        
        return results
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_report(
        self,
        correlations: List[AlphaCorrelation],
        apply_bonferroni: bool = True,
        n_tests: int = None
    ) -> str:
        """
        Generate human-readable report of findings.
        
        Args:
            correlations: List of significant correlations
            apply_bonferroni: Whether to apply multiple comparison correction
            n_tests: Total number of tests run (for Bonferroni)
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("ERP ALPHA DISCOVERY REPORT")
        lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        lines.append("")
        
        if not correlations:
            lines.append("No significant correlations found.")
            return "\n".join(lines)
        
        # Apply Bonferroni correction if requested
        if apply_bonferroni and n_tests:
            bonferroni_threshold = 0.05 / n_tests
            correlations = [c for c in correlations if c.pearson_p < bonferroni_threshold]
            lines.append(f"After Bonferroni correction ({n_tests} tests, α = {bonferroni_threshold:.4f}):")
            lines.append(f"Found {len(correlations)} significant correlation(s)")
        else:
            lines.append(f"Found {len(correlations)} significant correlation(s)")
        
        lines.append("")
        
        # Group by ERP signal
        by_signal: Dict[str, List[AlphaCorrelation]] = {}
        for corr in correlations:
            if corr.erp_signal not in by_signal:
                by_signal[corr.erp_signal] = []
            by_signal[corr.erp_signal].append(corr)
        
        for signal_name, corrs in by_signal.items():
            lines.append("-" * 50)
            lines.append(f"ERP SIGNAL: {signal_name}")
            lines.append("-" * 50)
            
            for c in corrs:
                direction = "↑" if c.direction == "positive" else "↓"
                lag_str = f"(Lag {c.lag_months}mo)" if c.lag_months > 0 else ""
                lines.append(
                    f"  {c.market_ticker:20} | r={c.pearson_r:+.3f} {direction} | "
                    f"p={c.pearson_p:.4f} | n={c.n_observations} {lag_str}"
                )
            lines.append("")
        
        # Summary
        lines.append("=" * 70)
        lines.append("INTERPRETATION")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Positive correlation (↑): When ERP signal rises, market tends to rise")
        lines.append("Negative correlation (↓): When ERP signal rises, market tends to fall")
        lines.append("Lag: ERP signal LEADS market by N months (predictive power)")
        lines.append("")
        lines.append("⚠️  DISCLAIMER: Correlation ≠ Causation")
        lines.append("These findings should be validated with domain knowledge and")
        lines.append("out-of-sample testing before use in trading strategies.")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================
    
    def run_full_scan(
        self,
        start_year: int = 2018,
        min_r: float = 0.5,
        max_p: float = 0.10,
        save_report: bool = True
    ) -> Tuple[List[AlphaCorrelation], str]:
        """
        Run complete ERP alpha scan.
        
        Returns:
            Tuple of (correlations list, report string)
        """
        print("=" * 60)
        print("  ERP ALPHA SCANNER")
        print("=" * 60)
        
        # Extract signals
        print("\n1. Extracting ERP signals...")
        signals = self.extract_all_signals(start_year)
        print(f"   Extracted {len(signals)} signals")
        
        # Fetch market data
        print("\n2. Fetching market data...")
        market_data = self.fetch_market_data(f"{start_year}-01-01")
        print(f"   Got {len(market_data.columns)} tickers, {len(market_data)} months")
        
        # Scan correlations
        print("\n3. Scanning correlations...")
        lags = [0, 1, 2, 3]
        n_tests = len(signals) * len(market_data.columns) * len(lags)
        correlations = self.scan_correlations(signals, market_data, lags, min_r, max_p)
        print(f"   Found {len(correlations)} significant correlations")
        
        # Generate report
        print("\n4. Generating report...")
        report = self.generate_report(correlations, apply_bonferroni=True, n_tests=n_tests)
        
        if save_report:
            from pathlib import Path
            report_path = Path(__file__).parent / "erp_alpha_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"   Report saved to: {report_path}")
        
        print("\n" + report)
        
        return correlations, report
    
    # =========================================================================
    # MOCK DATA (for testing without DB)
    # =========================================================================
    
    def _mock_signal(self, name: str) -> ERPSignal:
        """Generate mock signal for testing."""
        dates = pd.date_range(start='2018-01-01', end='2024-12-01', freq='MS')
        # Random walk with trend
        values = np.cumsum(np.random.randn(len(dates)) * 0.05) + np.linspace(0, 1, len(dates))
        values = (values - values.min()) / (values.max() - values.min()) * 1e6 + 5e5
        
        return ERPSignal(
            name=f"mock_{name}",
            description=f"Mock signal for {name}",
            frequency="monthly",
            data=pd.Series(values, index=dates),
            source_tables=["MOCK"]
        )
    
    def _mock_market_data(self) -> pd.DataFrame:
        """Generate mock market data for testing."""
        dates = pd.date_range(start='2018-01-01', end='2024-12-01', freq='MS')
        
        data = {}
        for ticker in ['corn', 'soybeans', 'materials_xlb']:
            returns = np.random.randn(len(dates)) * 0.05
            data[ticker] = returns
        
        return pd.DataFrame(data, index=dates)


# =============================================================================
# CLI & DEMO
# =============================================================================

def run_demo():
    """Demo with mock data (no database required)."""
    print("Running ERP Alpha Scanner Demo (Mock Data)")
    print("=" * 60)
    
    scanner = ERPAlphaScanner(cursor=None)  # No DB connection
    
    # This will use mock data
    correlations, report = scanner.run_full_scan(save_report=False)
    
    print("\nDemo complete!")
    print("To run with real data, ensure database connection is configured.")


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        run_demo()
    else:
        # Try real scan
        scanner = ERPAlphaScanner()
        if scanner.cursor:
            scanner.run_full_scan()
        else:
            print("No database connection. Run with --demo for mock data.")
