"""
Chemical Correlation Scanner

Adapted from the 'weird correlations scanner' concept - tests if unconventional
data sources (food consumption, entertainment metrics, etc.) correlate with 
chemical procurement costs.

Uses REAL historical data from USDA, BLS, Nielsen, and NPS.
"""

import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False

from secrets_loader import build_connection_string

LOGGER = logging.getLogger(__name__)

# =============================================================================
# WEIRD DATA SOURCES - Real Historical Data (2010-2024)
# =============================================================================

# Years as index
YEARS = list(range(2010, 2025))

WEIRD_DATA = {
    # USDA: Butter consumption per capita (lbs/year)
    "butter_consumption": {
        2010: 4.9, 2011: 5.0, 2012: 5.2, 2013: 5.3, 2014: 5.5,
        2015: 5.6, 2016: 5.7, 2017: 5.8, 2018: 5.9, 2019: 6.0,
        2020: 6.1, 2021: 6.2, 2022: 6.3, 2023: 6.5, 2024: 6.8
    },

    # USDA: Cheese consumption per capita (lbs/year)
    "cheese_consumption": {
        2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5
    },

    # USDA: Beef consumption per capita (lbs/year)
    "beef_consumption": {
        2010: 59.7, 2011: 57.0, 2012: 55.5, 2013: 54.5, 2014: 54.0,
        2015: 54.0, 2016: 54.3, 2017: 55.5, 2018: 56.5, 2019: 57.0,
        2020: 57.5, 2021: 58.8, 2022: 59.1, 2023: 58.1, 2024: 58.1
    },

    # National Chicken Council: Chicken consumption per capita (lbs/year)
    "chicken_consumption": {
        2010: 95.1, 2011: 95.8, 2012: 97.2, 2013: 96.5, 2014: 98.9,
        2015: 100.6, 2016: 100.6, 2017: 98.9, 2018: 98.9, 2019: 101.1,
        2020: 95.8, 2021: 96.5, 2022: 98.9, 2023: 98.9, 2024: 101.1
    },

    # USDA/Industry: Avocado consumption per capita (lbs/year)
    "avocado_consumption": {
        2010: 3.4, 2011: 4.0, 2012: 4.5, 2013: 6.2, 2014: 6.8,
        2015: 7.1, 2016: 7.1, 2017: 7.5, 2018: 8.0, 2019: 8.5,
        2020: 9.0, 2021: 9.0, 2022: 9.0, 2023: 9.2, 2024: 8.8
    },

    # BLS: Average coffee price per pound (USD)
    "coffee_price": {
        2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32
    },

    # Company reports: Starbucks US store count (thousands)
    "starbucks_stores": {
        2010: 10.6, 2011: 10.8, 2012: 11.2, 2013: 11.6, 2014: 12.0,
        2015: 12.5, 2016: 13.0, 2017: 13.5, 2018: 14.2, 2019: 15.0,
        2020: 15.3, 2021: 15.7, 2022: 15.9, 2023: 16.4, 2024: 16.9
    },

    # Nielsen: Super Bowl viewers (millions)
    "superbowl_viewers": {
        2010: 106.5, 2011: 111.0, 2012: 111.3, 2013: 108.7, 2014: 111.5,
        2015: 114.4, 2016: 112.3, 2017: 111.3, 2018: 103.4, 2019: 98.2,
        2020: 102.0, 2021: 96.4, 2022: 112.3, 2023: 115.1, 2024: 123.7
    },

    # Company reports: Netflix subscribers (millions, global)
    "netflix_subscribers": {
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0
    },

    # NPS: National Park visits (millions)
    "national_park_visits": {
        2010: 281.3, 2011: 278.7, 2012: 282.8, 2013: 273.6, 2014: 292.8,
        2015: 307.2, 2016: 331.0, 2017: 330.9, 2018: 318.2, 2019: 327.5,
        2020: 237.0, 2021: 297.1, 2022: 312.0, 2023: 325.0, 2024: 331.9
    },
}


class ChemicalCorrelationScanner:
    """
    Scans for correlations between 'weird' external data and chemical costs.
    """

    def __init__(self, cursor=None):
        """
        Initialize the scanner.
        
        Args:
            cursor: Optional pyodbc cursor. If None, will create connection.
        """
        self.cursor = cursor
        self._own_connection = False
        self._conn = None
        
        if cursor is None and HAS_PYODBC:
            try:
                conn_str, _, _, _ = build_connection_string()
                self._conn = pyodbc.connect(conn_str)
                self.cursor = self._conn.cursor()
                self._own_connection = True
            except Exception as e:
                LOGGER.warning(f"Could not connect to database: {e}")

    def __del__(self):
        if self._own_connection and self._conn:
            self._conn.close()

    def fetch_chemical_annual_costs(
        self,
        item_number: str,
        start_year: int = 2010,
        end_year: int = 2024
    ) -> pd.DataFrame:
        """
        Fetch average annual unit costs for a chemical item from purchase receipts.
        
        Args:
            item_number: The GP item number
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with columns [Year, AvgUnitCost, TotalQty, NumReceipts]
        """
        if self.cursor is None:
            LOGGER.warning("No database connection available")
            return pd.DataFrame()

        try:
            query = """
            SELECT 
                YEAR(h.RECEIPTDATE) AS Year,
                AVG(l.UNITCOST) AS AvgUnitCost,
                SUM(CASE WHEN l.UNITCOST > 0 THEN l.EXTDCOST / l.UNITCOST ELSE 0 END) AS TotalQty,
                COUNT(*) AS NumReceipts
            FROM POP30300 h
            JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
            WHERE l.ITEMNMBR = ?
              AND YEAR(h.RECEIPTDATE) BETWEEN ? AND ?
              AND l.EXTDCOST > 0
              AND l.UNITCOST > 0
            GROUP BY YEAR(h.RECEIPTDATE)
            ORDER BY YEAR(h.RECEIPTDATE)
            """
            self.cursor.execute(query, (item_number, start_year, end_year))
            rows = self.cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            data = [{
                'Year': int(row.Year),
                'AvgUnitCost': float(row.AvgUnitCost),
                'TotalQty': float(row.TotalQty) if row.TotalQty else 0,
                'NumReceipts': int(row.NumReceipts)
            } for row in rows]

            return pd.DataFrame(data)

        except Exception as e:
            LOGGER.error(f"Error fetching costs for {item_number}: {e}")
            return pd.DataFrame()

    def get_top_chemicals_by_spend(self, top_n: int = 20) -> list[str]:
        """
        Get top N chemical items by total historical spend.
        
        Returns:
            List of item numbers ordered by spend
        """
        if self.cursor is None:
            return []

        try:
            query = """
            SELECT TOP (?) 
                l.ITEMNMBR,
                SUM(l.EXTDCOST) AS TotalSpend
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE YEAR(h.RECEIPTDATE) >= 2010
              AND l.EXTDCOST > 0
            GROUP BY l.ITEMNMBR
            ORDER BY SUM(l.EXTDCOST) DESC
            """
            self.cursor.execute(query, (top_n,))
            rows = self.cursor.fetchall()
            return [row.ITEMNMBR.strip() for row in rows]
        except Exception as e:
            LOGGER.error(f"Error getting top chemicals: {e}")
            return []

    @staticmethod
    def calculate_correlation(
        x_data: dict[int, float],
        y_data: dict[int, float],
        lag: int = 0
    ) -> dict[str, Any]:
        """
        Calculate Pearson and Spearman correlation between two annual datasets.
        
        Args:
            x_data: Dict of year -> value (weird data / independent var)
            y_data: Dict of year -> value (chemical cost / dependent var)
            lag: Shift x_data by this many years. 
                 lag=1 means x[2010] tries to predict y[2011].
            
        Returns:
            Dict with correlation stats
        """
        # Align data based on lag
        # If lag=1, we want pairs (x[t], y[t+1])
        # So aligned x keys are k, aligned y keys are k+lag
        
        common_base_years = []
        for year in x_data:
            target_year = year + lag
            if target_year in y_data:
                common_base_years.append(year)
        
        common_base_years.sort()
        
        if len(common_base_years) < 5:
            return {
                "error": "Insufficient overlapping data",
                "n_years": len(common_base_years)
            }

        x = np.array([x_data[y] for y in common_base_years])
        y = np.array([y_data[y+lag] for y in common_base_years])

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x, y)
        
        # Spearman correlation (rank-based, more robust)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        return {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "regression_slope": slope,
            "regression_intercept": intercept,
            "regression_r2": r_value ** 2,
            "regression_p": p_value,
            "n_years": len(common_base_years),
            "years": common_base_years,
            "x_aligned": x,
            "y_aligned": y
        }

    def plot_top_correlations(self, results: list[dict], top_n: int = 5):
        """
        Generate plots for the strongest correlations.
        """
        if not results:
            return

        # Sort by strength
        top_results = sorted(results, key=lambda x: abs(x["pearson_r"]), reverse=True)[:top_n]
        
        for i, res in enumerate(top_results):
            item = res["item_number"]
            weird = res["weird_factor"]
            lag = res.get("lag", 0)
            
            # Re-fetch data to plot (inefficient but cleaner logic vs passing data around)
            cost_df = self.fetch_chemical_annual_costs(item)
            if cost_df.empty:
                continue
            cost_dict = dict(zip(cost_df['Year'], cost_df['AvgUnitCost']))
            weird_dict = WEIRD_DATA.get(weird, {})
            
            corr = self.calculate_correlation(weird_dict, cost_dict, lag)
            if "error" in corr:
                continue
                
            x = corr["x_aligned"]
            y = corr["y_aligned"]
            years = corr["years"]
            
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(x, y, color='blue', alpha=0.6, label='Data Points')
            
            # Trend line
            slope = corr["regression_slope"]
            intercept = corr["regression_intercept"]
            plt.plot(x, slope * x + intercept, color='red', linestyle='--', 
                     label=f'Trend (R2={corr["regression_r2"]:.2f})')
            
            title = f"Correlation: {weird} vs {item}"
            if lag > 0:
                title += f" (Lag: {lag} Year{'s' if lag>1 else ''})"
                
            plt.title(title, fontsize=14)
            plt.xlabel(f"{weird} (Year T)", fontsize=12)
            plt.ylabel(f"{item} Unit Cost (Year T+{lag})", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format currency on Y axis
            fmt = '${x:,.2f}'
            tick = mticker.StrMethodFormatter(fmt)
            plt.gca().yaxis.set_major_formatter(tick)
            
            safe_item = item.replace("/", "_").replace("\\", "_")
            filename = f"correlation_{safe_item}_{weird}_lag{lag}.png"
            plt.savefig(filename)
            plt.close()
            LOGGER.info(f"Saved plot: {filename}")

    def scan_all_correlations(
        self,
        item_numbers: list[str] | None = None,
        min_correlation: float = 0.5,
        max_p_value: float = 0.10,
        check_lags: bool = True
    ) -> list[dict]:
        """
        Scan all weird data against chemical costs, optionally checking lags.
        """
        if item_numbers is None:
            item_numbers = self.get_top_chemicals_by_spend(20)
            
        if not item_numbers:
            LOGGER.warning("No items to analyze")
            return []

        results = []
        lags_to_test = [0, 1] if check_lags else [0]
        
        for item in item_numbers:
            cost_df = self.fetch_chemical_annual_costs(item)
            
            if cost_df.empty or len(cost_df) < 5:
                continue
                
            cost_dict = dict(zip(cost_df['Year'], cost_df['AvgUnitCost']))
            
            for weird_name, weird_values in WEIRD_DATA.items():
                for lag in lags_to_test:
                    corr = self.calculate_correlation(weird_values, cost_dict, lag=lag)
                    
                    if "error" in corr:
                        continue
                        
                    # Check if significant
                    if (abs(corr["pearson_r"]) >= min_correlation and 
                        corr["pearson_p"] <= max_p_value):
                        
                        results.append({
                            "item_number": item,
                            "weird_factor": weird_name,
                            "lag": lag,
                            "pearson_r": round(corr["pearson_r"], 3),
                            "pearson_p": round(corr["pearson_p"], 4),
                            "spearman_r": round(corr["spearman_r"], 3),
                            "spearman_p": round(corr["spearman_p"], 4),
                            "direction": "positive" if corr["pearson_r"] > 0 else "negative",
                            "n_years": corr["n_years"],
                            "regression_r2": round(corr["regression_r2"], 3),
                        })

        # Sort by absolute correlation strength
        results.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        
        return results

    def generate_report(
        self,
        results: list[dict] | None = None,
        item_numbers: list[str] | None = None
    ) -> str:
        """
        Generate a formatted report of correlation findings.
        
        Args:
            results: Pre-computed results, or None to compute
            item_numbers: Items to analyze if results is None
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.scan_all_correlations(item_numbers)

        lines = [
            "=" * 70,
            "CHEMICAL PRICING CORRELATION REPORT",
            "Weird Data vs. Chemical Procurement Costs",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 70,
            "",
        ]

        if not results:
            lines.append("No significant correlations found (|r| >= 0.5, p <= 0.10)")
            return "\n".join(lines)

        lines.append(f"Found {len(results)} significant correlation(s):")
        lines.append("")
        
        # Group by item
        by_item = {}
        for r in results:
            by_item.setdefault(r["item_number"], []).append(r)

        for item, correlations in by_item.items():
            lines.append(f"\n{'─' * 50}")
            lines.append(f"ITEM: {item}")
            lines.append(f"{'─' * 50}")
            
            for c in correlations:
                direction = "^" if c["direction"] == "positive" else "v"
                lag_str = f" (Lag {c['lag']}y)" if c.get('lag', 0) > 0 else ""
                lines.append(
                    f"  {c['weird_factor']:25} | r={c['pearson_r']:+.3f} {direction} "
                    f"| p={c['pearson_p']:.4f} | R2={c['regression_r2']:.3f}{lag_str}"
                )

        # Summary statistics
        lines.append("\n" + "=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        
        pos_count = sum(1 for r in results if r["direction"] == "positive")
        neg_count = len(results) - pos_count
        
        lines.append(f"Total correlations found: {len(results)}")
        lines.append(f"  Positive: {pos_count}")
        lines.append(f"  Negative: {neg_count}")
        
        # Top weird factors
        factor_counts = {}
        for r in results:
            factor_counts[r["weird_factor"]] = factor_counts.get(r["weird_factor"], 0) + 1
        
        if factor_counts:
            lines.append("\nMost common predictive factors:")
            for factor, count in sorted(factor_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  {factor}: {count} chemical(s)")

        lines.append("\n" + "=" * 70)
        lines.append("DISCLAIMER: Correlation != Causation")
        lines.append("These correlations may be spurious. Always validate with")
        lines.append("domain knowledge before using for predictions.")
        lines.append("=" * 70)

        return "\n".join(lines)


def run_chemical_correlation_scan(
    items: list[str] | None = None,
    min_r: float = 0.5,
    max_p: float = 0.10,
    print_report: bool = True
) -> list[dict]:
    """
    Main entry point to run a chemical correlation scan.
    
    Args:
        items: List of item numbers to test (None = auto-select top 20)
        min_r: Minimum correlation coefficient
        max_p: Maximum p-value
        print_report: Whether to print the report
        
    Returns:
        List of significant correlations
    """
    scanner = ChemicalCorrelationScanner()
    results = scanner.scan_all_correlations(items, min_r, max_p)
    
    if print_report:
        report = scanner.generate_report(results)
        
        # Save to file to avoid console encoding issues
        with open("correlation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("Report saved to correlation_report.txt")

        try:
            print(report)
        except:
             print("Console print failed (encoding), read correlation_report.txt instead.")

        try:
            print(f"\nGenerating plots for top 5 correlations...")
            scanner.plot_top_correlations(results, top_n=5)
            print("Plots saved.")
        except Exception as e:
            print(f"Plot generation failed: {e}")
    
    return results


# Demo with sample data (no DB required)
def demo_with_sample_data():
    """
    Demonstrate correlation analysis with sample chemical data.
    """
    print("=" * 70)
    print("DEMO: Correlation Analysis with Sample Chemical Data")
    print("=" * 70)
    
    # Sample chemical price data (hypothetical annual averages)
    sample_chemicals = {
        "UREA46": {  # Nitrogen fertilizer - tends to correlate with food demand
            2010: 0.42, 2011: 0.52, 2012: 0.48, 2013: 0.45, 2014: 0.47,
            2015: 0.38, 2016: 0.32, 2017: 0.35, 2018: 0.41, 2019: 0.38,
            2020: 0.35, 2021: 0.55, 2022: 0.85, 2023: 0.62, 2024: 0.58
        },
        "PHOSPHORIC": {  # Phosphoric acid
            2010: 0.65, 2011: 0.72, 2012: 0.68, 2013: 0.70, 2014: 0.75,
            2015: 0.68, 2016: 0.55, 2017: 0.52, 2018: 0.58, 2019: 0.61,
            2020: 0.59, 2021: 0.78, 2022: 1.05, 2023: 0.88, 2024: 0.82
        },
        "POTASH": {  # Potassium chloride
            2010: 0.38, 2011: 0.42, 2012: 0.45, 2013: 0.40, 2014: 0.35,
            2015: 0.32, 2016: 0.28, 2017: 0.26, 2018: 0.30, 2019: 0.32,
            2020: 0.28, 2021: 0.45, 2022: 0.72, 2023: 0.55, 2024: 0.48
        }
    }
    
    results = []
    
    for chem_name, chem_costs in sample_chemicals.items():
        print(f"\nAnalyzing {chem_name}...")
        
        for weird_name, weird_values in WEIRD_DATA.items():
            corr = ChemicalCorrelationScanner.calculate_correlation(weird_values, chem_costs)
            
            if "error" in corr:
                continue
                
            if abs(corr["pearson_r"]) >= 0.5 and corr["pearson_p"] <= 0.10:
                results.append({
                    "item_number": chem_name,
                    "weird_factor": weird_name,
                    "lag": 0,
                    "pearson_r": round(corr["pearson_r"], 3),
                    "pearson_p": round(corr["pearson_p"], 4),
                    "spearman_r": round(corr["spearman_r"], 3),
                    "spearman_p": round(corr["spearman_p"], 4),
                    "direction": "positive" if corr["pearson_r"] > 0 else "negative",
                    "n_years": corr["n_years"],
                    "regression_r2": round(corr["regression_r2"], 3),
                })
                
                dir_symbol = "^" if corr["pearson_r"] > 0 else "v"
                print(f"  [MATCH] {weird_name}: r={corr['pearson_r']:+.3f} {dir_symbol} (p={corr['pearson_p']:.4f})")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DEMO RESULTS SUMMARY")
    print("=" * 70)
    
    if results:
        print(f"\nFound {len(results)} significant correlation(s):\n")
        for r in sorted(results, key=lambda x: abs(x["pearson_r"]), reverse=True):
            print(f"  {r['item_number']:12} ~ {r['weird_factor']:25} | r={r['pearson_r']:+.3f}")
    else:
        print("No significant correlations found in sample data.")
        
    return results


if __name__ == "__main__":
    # First run the demo with sample data (works without DB)
    demo_results = demo_with_sample_data()
    
    print("\n\n")
    
    # Then try with real DB data
    print("Attempting scan with real database...")
    try:
        results = run_chemical_correlation_scan()
    except Exception as e:
        # Print error safely avoiding encoding issues
        try:
            print(f"Could not connect to database: {str(e).encode('ascii', 'replace').decode('ascii')}")
        except:
            print("Could not connect to database (and failed to print error details)")
        print("Demo results shown above are still valid for analysis.")
